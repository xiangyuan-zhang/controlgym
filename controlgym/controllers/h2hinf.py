import numpy as np
import logging
from scipy.linalg import solve_discrete_are, solve, LinAlgError, LinAlgWarning
import warnings

warnings.filterwarnings("error")


def _h2hinf_gain(A: np.ndarray[float], B1: np.ndarray[float], B2: np.ndarray[float], 
                 C1: np.ndarray[float], D11: np.ndarray[float], D12: np.ndarray[float], gamma: float):
        """Private function to compute the gain matrices of the H2/Hinfinity controller
            using block algebraic Riccati equation (GARE).

        Args:
            A, B1, B2, C1, D11, D12: ndarray[float], system matrices.
            gamma: float, parameter that defines the robustness level of the H2/Hinfinity controller.
                when gamma -> infinity, the resulting controller is equivalent to the H2 controller.

        Returns:
            gain_act, gain_dis: ndarray[float], control and adversary gain matrices, respectively.
            If the GARE fails to find a solution, gain_act and gain_dis are set to None.
        """
        B_stacked = np.hstack((B2, B1))
        D1_stacked = np.hstack((D12, D11))
        X = np.hstack((D12.T @ D12, D12.T @ D11))
        Y = np.hstack((D11.T @ D12, D11.T @ D11 - (gamma**2) * np.identity((D11.T @ D11).shape[0])))
        R_stacked = np.vstack((X, Y))
        S = C1.T @ D1_stacked
        try:
            P = solve_discrete_are(A, B_stacked, C1.T @ C1, R_stacked, np.identity(A.shape[0]), S)
            gain_compact = solve(R_stacked + B_stacked.T @ P @ B_stacked, B_stacked.T @ P @ A + S.T)
        except Exception as e:
            if isinstance(e, ValueError):
                logging.warning(
                    str(e) + ". Input is ill-conditioned or gamma is too small!"
                )
            elif isinstance(e, LinAlgError) or isinstance(e, LinAlgWarning):
                # Handle the LinAlgWarning
                logging.warning(
                    "LinAlgError/Warning: solve_discrete_are failed to find a solution: input is ill-conditioned or gamma is too small!" +
                    "Another possible reason is gamma is too large such that numerical issues arise."
                )
            else:
                # Handle other exceptions
                logging.warning(
                    "An unexpected error occured when calling solve_discrete_are!"
                )
            gain_act, gain_dis = None, None
        else:
            gain_act = -gain_compact[: B2.shape[1], :]
            gain_dis = gain_compact[B2.shape[1] :, :]

        return gain_act, gain_dis

class H2Hinf:
    """
    ### Description

    This environment defines the H2/Hinfinity state-feedback controller for linear systems.
    The system dynamics is evolved based on the following discrete-time state-space model:
        state_{t+1} = A * state_t + B1 * disturbance_t + B2 * action_t
        output_t = C1 * state_t + D11 * disturbance_t + D12 * action_t
    The H2/Hinfinity controller is computed as:
        action_t = gain_act * state_t
        disturbance_t = gain_dis * state_t,
    where gain_act and gain_dis are the control and adversary gain matrices, respectively.

    ### Arguments
    For env_id in the following list:
    ["toy", "ac1", "ac2", "ac3", "ac4", "ac5", "ac6", "ac7", "ac8", "ac9", "ac10",
    "bdt1", "bdt2", "cbm", "cdp", "cm1", "cm2", "cm3", "cm4", "cm5",
    "dis1", "dis2", "dlr", "he1", "he2", "he3", "he4", "he5", "he6", "iss",
    "je1", "je2", "lah", "pas", "psm", "rea", "umv"]

    ```
    env = controlgym.make(env_id, **kwargs)
    controlgym.controllers.H2Hinf(env, gamma)
    ```

    Argument:
    [gamma]: float, parameter that defines the robustness level of the H2/Hinfinity controller.
            when gamma -> infinity, the resulting controller is equivalent to the H2 controller.
    """
    def __init__(self, env, gamma: float):
        self.env = env

        # check whether the problem is linear
        is_linear = self.env.category == "linear"
        
        assert is_linear and all(
            hasattr(self.env, attr) for attr in ["A", "B1", "B2", "C1", "D11", "D12"]
        ), "The environment is not linear or system matrices do not exist. H2Hinf is not applicable"

        A, B1, B2, C1, D11, D12 = (self.env.A, self.env.B1, self.env.B2, self.env.C1, self.env.D11, self.env.D12)

        # compute the control and adversary gain matrices
        self.gain_act, self.gain_dis = _h2hinf_gain(A, B1, B2, C1, D11, D12, gamma)

    def select_action(self, state: np.ndarray[float]):
        """Compute the H2/Hinfinity control input using state information.

        Args:
            state: ndarray[float], state information.

        Returns:
            action: ndarray[float], control input.
        """
        return self.gain_act @ state

    def select_disturbance(self, state: np.ndarray[float]):
        """Compute the H2/Hinfinity disturbance input using state information.

        Args:
            state: ndarray[float], state information.

        Returns:
            disturbance: ndarray[float], disturbance input.
        """
        return self.gain_dis @ state

    def run(self, state: np.ndarray[float] = None, seed: int = None):
        """Run a trajectory of the environment using the H2/Hinfinity controller,
            calculate the H2/Hinfinity cost, and save the state trajectory to env.state_traj.
            The trajectory is terminated when the environment returns a done signal (most likely
            due to the exceedance of the maximum number of steps: env.n_steps)
        Args:
            state: (optional ndarray[float]), an user-defined initial state.
            seed: (optional int), random seed for the environment.

        Returns:
            total_reward: float, the accumulated reward of the trajectory, 
                which is equal to the negative H2/Hinfinity cost.
        """
        # run evaluations with current policy
        _, info = self.env.reset(seed=seed, state=state)
        # run the simulated trajectory and calculate the h2 cost
        total_reward = 0
        state_traj = np.zeros((self.env.n_state, self.env.n_steps + 1))
        state_traj[:, 0] = info["state"]
        for t in range(self.env.n_steps):
            action = self.select_action(info["state"])
            disturbance = self.select_disturbance(info["state"])
            observation, reward, terminated, truncated, info = self.env.step(
                action, disturbance
            )
            state_traj[:, t + 1] = info["state"]
            if terminated or truncated:
                break
            total_reward += reward

        self.env.state_traj = state_traj
        return total_reward
