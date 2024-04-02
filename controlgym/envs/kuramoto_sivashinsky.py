import numpy as np
from controlgym.envs import PDE


class KuramotoSivashinskyEnv(PDE):
    """
    ### Description

    This environment models the Kuramoto–Sivashinsky equation, described by:
        du/dt =  -d^2u/dx^2 - u * du/dx - d^4u/dx^4

    ### Action Space, Observation Space, Rewards, Episode Termination

    See parent class `PDE` defined in pde.py for details.

    ### Arguments

    ```
    controlgym.make('kuramoto_sivashinsky')
    ```
    optional arguments:
    [n_steps]: the number of discrete-time steps. Default is 500.
    [domain_length]: domain length of PDE. Default is 32 * np.pi.
    [integration_time]: numerical integration time step. Default is 0.005.
    [sample_time]: each discrete-time step represents (ts) seconds. Default is 0.5.
    [process_noise_cov]: process noise covariance coefficient. Default is 0.0.
    [sensor_noise_cov]: sensor noise covariance coefficient. Default is 0.25.
    [target_state]: target state. Default is np.zeros(n_state).
    [init_amplitude_mean]: mean of initial amplitude. Default is 1.0.
    [init_amplitude_width]: width of initial amplitude. Default is 0.2.
    [n_state]: dimension of state vector. Default is 256.
    [n_observation]: dimension of observation vector. Default is 10.
    [n_action]: dimension of control vector. Default is 8.
    [control_sup_width]: control support width. Default is 0.1.
    [Q_weight]: weight of state tracking cost. Default is 1.0.
    [R_weight]: weight of control cost. Default is 1.0.
    [action_limit]: limit of action. Default is None.
    [observation_limit]: limit of observation. Default is None.
    [reward_limit]: limit of reward. Default is None.
    [seed]: random seed. Default is None.
    """

    def __init__(
        self,
        n_steps: int = 500,
        domain_length: float = 32 * np.pi,
        integration_time: float = 0.005,
        sample_time: float = 0.5,
        process_noise_cov: float = 0.0,
        sensor_noise_cov: float = 0.25,
        target_state: np.ndarray[float] = None,
        init_amplitude_mean: float = 1.0,
        init_amplitude_width: float = 0.2,
        n_state: int = 256,
        n_observation: int = 10,
        n_action: int = 8,
        control_sup_width: float = 0.1,
        Q_weight: float = 1.0,
        R_weight: float = 1.0,
        action_limit: float = None,
        observation_limit: float = None,
        reward_limit: float = None,
        seed: int = None,
    ):
        PDE.__init__(
            self,
            id="kuramoto_sivashinsky",
            n_steps=n_steps,
            domain_length=domain_length,
            integration_time=integration_time,
            sample_time=sample_time,
            process_noise_cov=process_noise_cov,
            sensor_noise_cov=sensor_noise_cov,
            target_state=target_state,
            n_state=n_state,
            n_observation=n_observation,
            n_action=n_action,
            control_sup_width=control_sup_width,
            Q_weight=Q_weight,
            R_weight=R_weight,
            action_limit=action_limit,
            observation_limit=observation_limit,
            reward_limit=reward_limit,
            seed=seed,
        )

        # initial state parameters
        self.init_amplitude_mean = init_amplitude_mean
        self.init_amplitude_width = init_amplitude_width
        self.reset()

    def select_init_state(self, init_amplitude=None):
        """Function to select the initial state of the PDE."""
        if init_amplitude is None:
            random_amplitude = self.rng.uniform(
                -0.5 * self.init_amplitude_width, 0.5 * self.init_amplitude_width
            )
            init_amplitude = self.init_amplitude_mean + random_amplitude
        init_state = (
            init_amplitude
            * np.cos(2 * np.pi * self.domain_coordinates / self.domain_length)
            * np.sin(12 * np.pi * self.domain_coordinates / self.domain_length)
        )
        return init_state

    def _compute_fourier_linear_op(self):
        """Private function to compute the linear operator of the PDE in Fourier space.

        Args:
            None.

        Returns:
            Linear operator of the PDE in Fourier space.
        """
        fourier_linear_op = self.domain_wavenumbers**2 - self.domain_wavenumbers**4
        return fourier_linear_op

    def _compute_fourier_nonlinear_op(self):
        """Private function to compute the nonlinear operator of the PDE in Fourier space.

        Args:
            None.

        Returns:
            A function that computes the nonlinear operator of the PDE in Fourier space.
        """

        def fourier_nonlinear_op(state_fourier, action):
            # aa_state is the anti-aliased state; meaning the state evaluated in
            # physical space on a grid with 3/2 times more points
            aa_factor = 3 / 2
            aa_state = aa_factor * np.fft.irfft(
                state_fourier, n=int(self.n_state * aa_factor)
            )
            right_hand_side = (
                -0.5j
                * self.domain_wavenumbers
                * (1 / aa_factor)
                * np.fft.rfft(aa_state**2)[0 : int(self.n_state / 2) + 1]
            ) + (np.fft.rfft(self.control_sup, axis=0) @ action)
            return right_hand_side

        return fourier_nonlinear_op

    def get_params_asdict(self):
        """Save the extra environment parameters as a dictionary.

        Args:
            None.

        Returns:
            a dictionary containing the parameters of the pde environment + extra parameters.
        """
        pde_dict = super().get_params_asdict()
        extra_data = {
            "init_amplitude_mean": self.init_amplitude_mean,
            "init_amplitude_width": self.init_amplitude_width,
        }
        return {**pde_dict, **extra_data}
