import numpy as np
from scipy.signal import cont2discrete

def ft_matrix(n_state):
    """
    Computes the DFT matrix of size N with 'backward' normalization mode in numpy.fft.fft

    Args:
        n_state: number of discretization points

    Returns:
        DFT: discrete Fourier transform matrix
    """
    DFT = np.zeros((n_state, n_state), dtype=np.complex128)
    omega = np.exp(-2j * np.pi / n_state)  # Twiddle factor
    for i in range(n_state):
        for j in range(n_state):
            DFT[i, j] = omega ** (i * j)
    return DFT


def ift_matrix(n_state):
    """
    Computes the IDFT matrix of size N with 'backward' normalization mode in numpy.fft.fft

    Args:
        n_state: number of discretization points

    Returns:
        IDFT: inverse discrete Fourier transform matrix
    """
    DFT = ft_matrix(n_state)
    IDFT = np.conjugate(DFT.T) / n_state
    return IDFT

def c2d(A_cont, B1_cont, B2_cont, C1_cont, D11_cont, D12_cont, sample_time):
    """Discretize the continuous-time system.

        Args:
            A_cont: The A matrix of the continuous-time system.
            B1_cont: The B1 matrix of the continuous-time system.
            B2_cont: The B2 matrix of the continuous-time system.
            C1_cont: The C1 matrix of the continuous-time system.
            D11_cont: The D11 matrix of the continuous-time system.
            D12_cont: The D12 matrix of the continuous-time system.
            sample_time: The sample time of the discrete-time system.

        Returns:
            A_disc: The A matrix of the discrete-time system.
            B1_disc: The B1 matrix of the discrete-time system.
            B2_disc: The B2 matrix of the discrete-time system.
            C1_disc: The C1 matrix of the discrete-time system.
            D11_disc: The D11 matrix of the discrete-time system.
            D12_disc: The D12 matrix of the discrete-time system.
    """
    # Stack B1 and B2 horizontally
    B_cont_stacked = np.hstack((B1_cont, B2_cont))
    # Stack D11 and D12 horizontally
    D1_cont_stacked = np.hstack((D11_cont, D12_cont))
    # Discretize the system
    A_disc, B_disc_stacked, C1_disc, D1_disc_stacked, _ = cont2discrete(
        (A_cont, B_cont_stacked, C1_cont, D1_cont_stacked), sample_time, method="zoh"
    )
    # Split the stack of B1 and B2
    B1_disc = B_disc_stacked[:, : B1_cont.shape[1]]
    B2_disc = B_disc_stacked[:, B1_cont.shape[1] :]
    # Split the stack of D11 and D12
    D11_disc = D1_disc_stacked[:, : D11_cont.shape[1]]
    D12_disc = D1_disc_stacked[:, D11_cont.shape[1] :]

    return A_disc, B1_disc, B2_disc, C1_disc, D11_disc, D12_disc

