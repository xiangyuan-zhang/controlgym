import numpy as np
from numpy.linalg import inv
import logging
from scipy.linalg import solve_discrete_are, solve, LinAlgError


def _lqr_gain(A: np.ndarray[float], B: np.ndarray[float], Q: np.ndarray[float], 
              R: np.ndarray[float], S: np.ndarray[float]):
    """Private function to compute the gain matrices of the LQR controller
        using algebraic Riccati equation (ARE).

    Args:
        A, B, Q, R, S: ndarray[float], system matrices.

    Returns:
        gain_lqr: ndarray[float], control gain matrix for regulation.
        gain_lqt: ndarray[float], additional control gain matrix for tracking.
        If the ARE fails to find a solution, gain is set to None.
    """
    try:
        P = solve_discrete_are(A, B, Q, R, e=None, s=S)
    except LinAlgError:
        # handle the exceptional case where solve_discrete_are fails
        logging.warning("solve_discrete_are failed to find a finite solution")
        gain_lqr = gain_lqt = None
    else:
        gain_lqr = -solve(R + B.T @ P @ B, B.T @ P @ A + S.T)
        # compute the LQ tracking gain for PDE envs, where S = 0.
        gain_lqt = inv(R + B.T @ P @ B) @ B.T @ inv(np.identity(A.shape[0]) - (A + B @ gain_lqr).T) @ Q
    return gain_lqr, gain_lqt


class LQR:
    """
    ### Description

    This environment defines the LQR state-feedback controller for linear systems.
    The system dynamics is evolved based on the following discrete-time state-space model:
        state_{t+1} = A * state_t + B2 * action_t + B1 * noise_t
        output_t = C1 * state_t + D12 * action_t + D11 * noise_t
        noise_t = N(0, noise_cov * I)
    The LQR controller is computed as:
        action_t = gain_lqr * state_t + gain_lqt * target_state 
    where gain_lqr is the control gain matrix.

    ### Arguments
    For env_id in the following list:
    ["toy", "ac1", "ac2", "ac3", "ac4", "ac5", "ac6", "ac7", "ac8", "ac9", "ac10",
    "bdt1", "bdt2", "cbm", "cdp", "cm1", "cm2", "cm3", "cm4", "cm5",
    "dis1", "dis2", "dlr", "he1", "he2", "he3", "he4", "he5", "he6", "iss",
    "je1", "je2", "lah", "pas", "psm", "rea", "umv", "convection_diffusion_reaction",
    "wave", "schrodinger"]

    ```
    env = controlgym.make(env_id, **kwargs)
    controlgym.controllers.LQR(env)
    ```

    Argument:
        None.
    """

    def __init__(self, env):
        self.env = env

        # check whether the problem is linear
        is_linear = self.env.category == "linear" or self.env.id in [
            "convection_diffusion_reaction",
            "wave",
            "schrodinger",
        ]
        assert is_linear and all(
            hasattr(self.env, attr) for attr in ["A", "B2"]
        ), "The environment is not linear or system matrices do not exist. LQR is not applicable"

        A, B2 = self.env.A, self.env.B2
        Q = self.env.Q if hasattr(self.env, "Q") else np.identity(self.env.n_state)
        R = self.env.R if hasattr(self.env, "R") else np.identity(self.env.n_action)
        S = self.env.S if hasattr(self.env, "S") else np.zeros((self.env.n_state, self.env.n_action))

        # compute the LQR gain
        self.gain_lqr, self.gain_lqt = _lqr_gain(A, B2, Q, R, S)

    def select_action(self, state: np.ndarray[float]):
        """Compute the LQR control input using state information.

        Args:
            state: ndarray[float], state information.

        Returns:
            action: ndarray[float], control input.
        """
        # PDE settings where the objective is to trak a target state
        if hasattr(self.env, "target_state"):
            return self.gain_lqr @ state + self.gain_lqt @ self.env.target_state
        else:
            return self.gain_lqr @ state

    def run(self, state: np.ndarray[float] = None, seed: int = None):
        """Run a trajectory of the environment using the LQR controller,
            calculate the H2 cost, and save the state trajectory to env.state_traj.
            The trajectory is terminated when the environment returns a done signal (most likely
            due to the exceedance of the maximum number of steps: env.n_steps)
        Args:
            state: (optional ndarray[float]), an user-defined initial state.
            seed: (optional int), random seed for the environment.

        Returns:
            total_reward: float, the accumulated reward of the trajectory,
                which is equal to the negative H2 cost.
        """
        # reset the environment
        _, info = self.env.reset(seed=seed, state=state)
        # run the simulated trajectory and calculate the h2 cost
        total_reward = 0
        state_traj = np.zeros((self.env.n_state, self.env.n_steps + 1))
        state_traj[:, 0] = info["state"]

        for t in range(self.env.n_steps):
            action = self.select_action(info["state"])
            observation, reward, terminated, truncated, info = self.env.step(action)
            state_traj[:, t + 1] = info["state"]
            if terminated or truncated:
                break
            total_reward += reward

        self.env.state_traj = state_traj
        return total_reward
