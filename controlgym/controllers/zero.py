import numpy as np

class Zero:
    """
    ### Description

    This environment implements the zero controller. 

    ### Arguments
    For env_id in the following list:
    ["toy", "ac1", "ac2", "ac3", "ac4", "ac5", "ac6", "ac7", "ac8", "ac9", "ac10",
    "bdt1", "bdt2", "cbm", "cdp", "cm1", "cm2", "cm3", "cm4", "cm5", "dis1", "dis2", 
    "dlr", "he1", "he2", "he3", "he4", "he5", "he6", "iss", "je1", "je2", "lah", 
    "pas", "psm", "rea", "umv", "allen_cahn", "burgers", "cahn_hilliard", 
    "convection_diffusion_reaction", "fisher", "ginzburg_landau",
    "korteweg_de_vries", "kuramoto_sivashinsky", "schrodinger", "wave"]

    ```
    env = controlgym.make(env_id, **kwargs)
    controlgym.controllers.Zero(env)
    ```

    Argument:
        None.
    """
    def __init__(self, env):
        self.env = env

    def select_action(self):
        """Returns the zero control input."""
        return np.zeros(self.env.n_action)

    def run(self, state=None, seed=None):
        """Run a trajectory of the environment using zero control,
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
        state_traj = np.zeros((self.env.n_state, self.env.n_steps+1))
        state_traj[:, 0] = info['state']
        for t in range(self.env.n_steps):
            action = self.select_action()
            observation, reward, terminated, truncated, info = self.env.step(action)

            state_traj[:, t+1] = info["state"]
            if terminated or truncated:
                break
            total_reward += reward

        self.env.state_traj = state_traj
        return total_reward
