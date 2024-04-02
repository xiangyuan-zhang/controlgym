import numpy as np
import os
import torch
from torch import nn


class ActorNetwork(nn.Module):
    """ Defines the actor network for the PPO controller. """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, device: torch.device = torch.device("cpu")):
        super(ActorNetwork, self).__init__()

        self.device = device
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, input: np.ndarray[float]):
        input = torch.FloatTensor(input).to(self.device)
        output = self.relu(self.fc1(input))
        output = self.relu(self.fc2(output))
        output = self.fc3(output)

        return output


class CriticNetwork(nn.Module):
    """ Defines the critic network for the PPO controller. """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, device: torch.device = torch.device("cpu")):
        super(CriticNetwork, self).__init__()

        self.device = device
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, input: np.ndarray[float]):
        input = torch.FloatTensor(input).to(self.device)
        output = self.relu(self.fc1(input))
        output = self.relu(self.fc2(output))
        output = self.fc3(output)

        return output


class PPO:
    """
    ### Description

    This environment defines the PPO observation-feedback controller for general systems.

    ### Arguments
    For env_id in the following list:
    ["toy", "ac1", "ac2", "ac3", "ac4", "ac5", "ac6", "ac7", "ac8", "ac9", "ac10",
    "bdt1", "bdt2", "cbm", "cdp", "cm1", "cm2", "cm3", "cm4", "cm5", 
    "dis1", "dis2", "dlr", "he1", "he2", "he3", "he4", "he5", "he6", "iss",
    "je1", "je2", "lah", "pas", "psm", "rea", "umv", "allen_cahn", "burgers",
    "cahn_hilliard", "convection_diffusion_reaction", "fisher", "ginzburg_landau",
    "korteweg_de_vries", "kuramoto_sivashinsky", "schrodinger", "wave"]

    ```
    env = controlgym.make(env_id, **kwargs)
    controlgym.controllers.PPO(env)
    ```

    Argument:
        env: object, controlgym environment object.
        actor_hidden_dim: int, hidden dimension of the actor network.
        critic_hidden_dim: int, hidden dimension of the critic network.
        lr: float, learning rate.
        device: torch.device, device to run the controller.
    """
    def __init__(self, env, actor_hidden_dim: int = 64, critic_hidden_dim: int = 64, 
                 lr: int = 1e-4, discount_factor: float = 0.99, device: torch.device = torch.device("cpu")):
        self.env = env
        self.actor_hidden_dim = actor_hidden_dim
        self.critic_hidden_dim = critic_hidden_dim
        self.device = device
        self.discount_factor = discount_factor

        self.actor = ActorNetwork(
            self.env.n_observation, self.actor_hidden_dim, self.env.n_action, device=device
        ).to(device)
        self.critic = CriticNetwork(
            self.env.n_observation, self.critic_hidden_dim, 1, device=device
        ).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

    def select_action(self, observation: np.ndarray[float], cov_param: float):
        """Select an action using the PPO controller.
        Args:
            observation: ndarray[float], current observation information.
            cov_param: float, the covariance parameter of the multivariate Gaussian distribution.

        Returns:
            action: ndarray[float], the selected action.
            log_prob: float, the log probability of the selected action.
        """
        mean = self.actor(observation)
        if cov_param == 0.0:
            action = mean
            return torch.Tensor.cpu(action).detach().numpy()
        else:
            cov = torch.diag(torch.ones(self.env.n_action) * cov_param).to(self.device)
            dist = torch.distributions.MultivariateNormal(mean, cov)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return torch.Tensor.cpu(action).detach().numpy(), torch.Tensor.cpu(log_prob).detach()

    def train(self, num_train_iter: int = 50, num_episodes_per_iter: int = 20, episode_length: int = 20, 
              sgd_epoch_num: int = 4, mini_batch_size: int = 5, clip: float = 0.2, cov_param: float = 0.05):
        """Training loop for the PPO controller.
        Args:
            num_train_iter: int, number of training iterations.
            num_episodes_per_iter: int, number of episodes per training iteration.
            episode_length: int, maximum episode length.
            sgd_epoch_num: int, number of epochs for stochastic gradient descent.
            mini_batch_size: int, mini batch size for stochastic gradient descent.
            clip: float, clipping parameter for PPO.
            cov_param: float, the covariance parameter of the multivariate Gaussian distribution.

        Returns:
            None.
        """
        for iteration in range(num_train_iter):
            observations = []
            actions = []
            rewards = []
            log_probs = []

            for episode in range(num_episodes_per_iter):
                observation, info = self.env.reset()
                episode_rewards = []

                for step in range(episode_length):
                    action, log_prob = self.select_action(observation, cov_param=cov_param)
                    next_obs, reward, terminated, truncated, info = self.env.step(action)
                    if terminated or truncated:
                        break

                    observations.append(observation)
                    episode_rewards.append(reward)
                    log_probs.append(log_prob)
                    actions.append(action)
                    observation = next_obs

                rewards.append(episode_rewards)

            observations = torch.FloatTensor(np.array(observations))
            actions = torch.FloatTensor(np.array(actions)).to(self.device)
            log_probs = torch.FloatTensor(np.array(log_probs)).to(self.device)

            average_rewards = np.mean([np.sum(episode_rewards) for episode_rewards in rewards])
            print("Iteration: ", iteration, ", Average Rewards: ", average_rewards)
            returns = self._calc_GAE(rewards)

            values = self.critic(observations).squeeze()
            advantage = returns - values.detach()
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            for epoch in range(sgd_epoch_num):
                batch_size = observations.size(0)

                for mini_batch in range(batch_size // mini_batch_size):
                    ids = np.random.randint(0, batch_size, mini_batch_size)

                    values = self.critic(observations[ids]).squeeze()
                    mean = self.actor(observations[ids])
                    cov = torch.diag(torch.ones(self.env.n_action) * cov_param).to(self.device)
                    dist = torch.distributions.MultivariateNormal(mean, cov)

                    log_probs_new = dist.log_prob(actions[ids])
                    entropy = dist.entropy().mean()

                    ratios = (log_probs_new - log_probs[ids]).exp()

                    surr1 = ratios * advantage[ids]
                    surr2 = torch.clamp(ratios, 1 - clip, 1 + clip) * advantage[ids]
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = (returns[ids] - values).pow(2).mean()

                    loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

    def _calc_GAE(self, rewards: list):
        """Private function to calculate the generalized advantage estimation (GAE).
        Args:
            rewards: list, a list of episode rewards.

        Returns:
            returns: torch.FloatTensor, the GAE.
        """
        returns = []
        for episode_rewards in reversed(rewards):
            discounted_return = 0.0
            for reward in reversed(episode_rewards):
                discounted_return = reward + discounted_return * self.discount_factor
                returns.insert(0, discounted_return)

        returns = torch.FloatTensor(returns).to(self.device)
        return returns

    def run(self, state: np.ndarray[float] = None, seed: int = None):
        """Run a trajectory of the environment using the PPO controller,
            calculate the total reward, and save the state trajectory to env.state_traj.
            The trajectory is terminated when the environment returns a done signal (most likely
            due to the exceedance of the maximum number of steps: env.n_steps)
        Args:
            state: (optional ndarray[float]), an user-defined initial state.
            seed: (optional int), random seed for the environment.

        Returns:
            total_reward: float, the accumulated reward of the trajectory.
        """
        # reset the environment
        observation, info = self.env.reset(seed=seed, state=state)
        if seed is not None:
            torch.manual_seed(seed=seed)
        # run the simulated trajectory and calculate the h2 cost
        total_reward = 0
        state_traj = np.zeros((self.env.n_state, self.env.n_steps + 1))
        state_traj[:, 0] = info["state"]
        for t in range(self.env.n_steps):
            action = self.select_action(observation, cov_param=0.0)
            observation, reward, terminated, truncated, info = self.env.step(action)
            state_traj[:, t + 1] = info["state"]
            if terminated or truncated:
                break
            total_reward += reward
        self.env.state_traj = state_traj
        return total_reward

    def save(self, test_dir: str = None):
        """Save the PPO parameters to the test directory.
        Args:
            test_dir: str, directory to save the trained parameters.

        Returns:
            None.
        """
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"{test_dir} does not exist")

        ppo_params = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(ppo_params, os.path.join(test_dir, "ppo_params.pt"))

    def load(self, test_dir: str = None):
        """Load the PPO parameters from the test directory.
        Args:
            test_dir: str, directory to load the trained parameters.

        Returns:
            None.
        """
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"{test_dir} does not exist")

        ppo_params = torch.load(os.path.join(test_dir, "ppo_params.pt"))
        self.actor.load_state_dict(ppo_params["actor"])
        self.critic.load_state_dict(ppo_params["critic"])
        self.optimizer.load_state_dict(ppo_params["optimizer"])
