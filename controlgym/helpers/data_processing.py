import os
from datetime import datetime
from scipy.io import savemat, loadmat
from controlgym.helpers.plotting import _plot_pde, _plot_linear, _plot_coupled_pde, _plot_linearPDE_eigen


def save(controller, path: str = None, temp_save: bool = True, display: bool = False, 
         contour: bool = True, surface3d: bool = False, dpi: int = 100):
    """Save the test data of the enviroment, controller, and figures.
        Args:
            controller: object, the tested controller to be saved.
            temp_save: bool, whether to save the test data in a temporary folder.
            display: bool, whether to display the images using plt.show().
            contour: bool, whether to plot and save the contour of the state trajectory (for PDE envs only).
            surface3d: bool, whether to plot and save the 3D surface of the state trajectory (for PDE envs only).
            dpi: int, the resolution of the saved figures.

            ** if temp_save is True, only the state trajectory will be saved, and the controller policy
                and environment parameters will not be saved.
            If you need to save the controller policy and environment parameters, set temp_save to False.
            And the test data will be saved in a new folder named after the controller name and the current time stamp.
        Returns:
            test_dir_path: str, the path of the saved test data.
    """
    controller_name = controller.__class__.__name__
    time_now = datetime.now().strftime("%m%d%H%M%S")
    
    folder_path = path or ""
    env_dir_path = os.path.join(folder_path, "test_data", controller.env.id)

    if temp_save:
        # overwrite the temp folder
        test_dir_path = os.path.join(env_dir_path, "temp")
    else:
        test_dir_path = os.path.join(env_dir_path, controller_name + "_" + time_now)

    # create a new test folder to save the test data
    os.makedirs(test_dir_path, exist_ok=True)

    # save the environment parameters and control policy only if temp_save is False
    if not temp_save:
        savemat(
            os.path.join(test_dir_path, "env_params.mat"),
            controller.env.get_params_asdict(),
        )
        if hasattr(controller, "save"):
            controller.save(test_dir_path)

    # plot the state trajectory
    if controller.env.id == "convection_diffusion_reaction":
        _plot_linearPDE_eigen(controller.env, test_dir_path, display=False, dpi=dpi)
    if controller.env.id in ["wave", "schrodinger"]:
        _plot_coupled_pde(controller.env, test_dir_path, display, contour, surface3d, dpi)
        _plot_linearPDE_eigen(controller.env, test_dir_path, display=False, dpi=dpi)
    elif controller.env.category == "pde":
        _plot_pde(controller.env, test_dir_path, display, contour, surface3d, dpi)
    elif controller.env.category == "linear":
        _plot_linear(controller.env, test_dir_path, display, dpi)

    return test_dir_path


def load_env_params(dir_path: str):
    """Load the enviroment parameter file from a specific test folder
        Args:
            dir_path: str, the path of the test folder.

        Returns:
            env_params: dict, a dictionary that contains the environment parameters.
    """
    # check whether the test folder exists
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"{dir_path} does not exist")
    return loadmat(os.path.join(dir_path, "env_params.mat"))
    
