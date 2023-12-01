import numpy as np
import logging
from scipy.linalg import solve_discrete_are, solve, inv, LinAlgError


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

def _kf_gain(A: np.ndarray[float], C: np.ndarray[float], W: np.ndarray[float], V: np.ndarray[float]):
    """Private function to compute the gain matrices of the Kalman filter
            using algebraic Riccati equation (ARE).

        Args:
            A, C, W, V: ndarray[float], system matrices and noise statistics.

        Returns:
            gain: ndarray[float], filter gain matrix.
            If the ARE fails to find a solution, gain is set to None.
        """
    try:
        Sigma = solve_discrete_are(A.T, C.T, W, V)
    except LinAlgError:
        # handle the exceptional case where solve_discrete_are fails
        logging.warning(
            "solve_discrete_are failed to find a finite solution for KF"
        )
        gain = None
    else:
        gain = A @ Sigma @ C.T @ inv(C @ Sigma @ C.T + V)
    return gain


class LQG:
    """
    ### Description

    This environment defines the LQG observation-feedback controller for linear systems.
    The system dynamics is evolved based on the following discrete-time state-space model:
        (linear control envs):
            state_{t+1} = A * state_t + B2 * action_t + B1 * noise_t
            observation_t = C * state_t + D21 * noise_t
            noise_t = N(0, noise_cov * I)
        (Linear PDE envs):
            state_{t+1} = A * state_t + B2 * action_t + process_noise_t
            observation_t = C * state_t + sensor_noise_t
            process_noise_t = N(0, process_noise_cov * I)
            sensor_noise_t = N(0, sensor_noise_cov * I)
        
    The LQG controller is computed as:
        action_t = gain_lqr * est_state_t + gain_lqt * target_state
        est_state_{t+1} = (A - gain_kf * C) * est_state_t + B2 * action_t
                            + gain_kf * observation_t
    where gain_lqr and gain_lqt are the control gain matrices.

    ### Arguments
    For env_id in the following list:
    ["toy", "ac1", "ac2", "ac3", "ac4", "ac5", "ac6", "ac7", "ac8", "ac9", "ac10",
    "bdt1", "bdt2", "cbm", "cdp", "cm1", "cm2", "cm3", "cm4", "cm5", 
    "dis1", "dis2", "dlr", "he1", "he2", "he3", "he4", "he5", "he6", "iss",
    "je1", "je2", "lah", "pas", "psm", "rea", "umv", "convection_diffusion_reaction",
    "wave", "schrodinger"]

    ```
    env = controlgym.make(env_id, **kwargs)
    controlgym.controllers.LQG(env)
    ```

    Argument:
        None.
    """
    def __init__(self, env):
        self.env = env

        # check whether the problem is linear
        is_linear = self.env.category == "linear" or self.env.id in [
            "convection_diffusion_reaction", "wave", "schrodinger"]
        assert is_linear and all(
            hasattr(self.env, attr) for attr in ["A", "B2", "C"]
        ), "The environment is not linear or system matrices do not exist. LQG is not applicable"

        A, B2, C = self.env.A, self.env.B2, self.env.C
        Q = self.env.Q if hasattr(self.env, "Q") else np.identity(self.env.n_state)
        R = self.env.R if hasattr(self.env, "R") else np.identity(self.env.n_action)
        S = (
            self.env.S
            if hasattr(self.env, "S")
            else np.zeros((self.env.n_state, self.env.n_action))
        )

        # PDE environments
        if hasattr(self.env, "sensor_noise_cov"):
            assert (
                self.env.sensor_noise_cov > 0
            ), "Sensor noise covariance must be positive"
            V = self.env.sensor_noise_cov * np.identity(self.env.n_observation)
        # Linear control environments
        elif hasattr(self.env, "D21"):
            V = self.env.D21 @ self.env.D21.T
            if min(np.linalg.eigvals(V)) <= 0:
                V = self.env.noise_cov * np.identity(self.env.n_observation)
        # Other environments
        else:
            V = 0.1 * np.identity(self.env.n_observation)

        # PDE environments
        if hasattr(self.env, "process_noise_cov"):
            W = self.env.process_noise_cov * np.identity(self.env.n_state)
        # Linear control environments
        elif hasattr(self.env, "B1"):
            W = self.env.B1 @ self.env.B1.T
        # Other environments
        else:
            W = 0.1 * np.identity(self.env.n_state)

        # compute the LQR and KF gains
        self.gain_lqr, self.gain_lqt = _lqr_gain(A, B2, Q, R, S)
        self.gain_kf = _kf_gain(A, C, W, V)

        self.traj = None

    def select_action(self, est_state: np.ndarray[float]):
        """Compute the LQG control input using estimated state.

        Args:
            est_state: ndarray[float], estimate state from Kalman filtering.

        Returns:
            action: ndarray[float], control input.
        """
        if hasattr(self.env, "target_state"):
            return self.gain_lqr @ est_state + self.gain_lqt @ self.env.target_state
        else:
            return self.gain_lqr @ est_state

    def evolve_state_estimate(self, est_state: np.ndarray[float], action: np.ndarray[float], observation: np.ndarray[float]):
        """Update the state estimate using the Kalman filter with new observation and control input.

        Args:
            est_state: ndarray[float], current estimated state from Kalman filtering.
            action: ndarray[float], current control input.
            observation: ndarray[float], current observation.

        Returns:
            est_state: ndarray[float], updated state estimate.
        """
        return (self.env.A - self.gain_kf @ self.env.C) @ est_state + self.env.B2 @ action + self.gain_kf @ observation

    def run(self, state: np.ndarray[float] = None, seed: int = None):
        """Run a trajectory of the environment using the LQG controller,
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

        est_state = info["state"]
        for t in range(self.env.n_steps):
            action = self.select_action(est_state)
            observation, reward, terminated, truncated, info = self.env.step(action)
            state_traj[:, t + 1] = info["state"]
            if terminated or truncated:
                break
            est_state = self.evolve_state_estimate(est_state, action, observation)
            total_reward += reward

        self.env.state_traj = state_traj
        return total_reward
