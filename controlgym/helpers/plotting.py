import numpy as np
import matplotlib.pyplot as plt
import os


def _plot_pde(env, save_dir: str = None, display: bool = False, contour: bool = True, 
              surface3d: bool = True, dpi: int = 100):
    """Plot the state trajectory of a PDE task.
        Args:
            env: object, the tested linear control environment.
            save_dir: str, the path of the test folder.
            discrete_time: bool, whether to plot the state trajectory in discrete time.
            contour: bool, whether to plot and save the contour of the state trajectory.
            surface3d: bool, whether to plot and save the 3D surface of the state trajectory.
            dpi: int, the resolution of the saved figures.

        Returns:
            None.
    """
    # check whether save_dir is valid
    if not os.path.exists(save_dir):
        # handled by the helpers/data_processing/save
        raise FileNotFoundError(f"{save_dir} does not exist")
    if env.state_traj is None:
        raise ValueError("State_traj is empty. Please run the controller first.")
    
    time = np.linspace(0, env.n_steps * env.sample_time, env.n_steps + 1)
    contour_cmap_dict = {"allen_cahn":"Spectral_r", "burgers":"coolwarm", "kuramoto_sivashinsky":"viridis", 
                         "cahn_hilliard":"Spectral_r", "convection_diffusion_reaction":"coolwarm", "fisher":"Spectral_r", 
                         "ginzburg_landau":"Spectral_r", "korteweg_de_vries":"Spectral_r"}
    sparcity_constant_dict = {"allen_cahn":10, "burgers":5, "kuramoto_sivashinsky":50, 
                             "cahn_hilliard":5, "convection_diffusion_reaction":5, 
                             "fisher":10, "ginzburg_landau":10, "korteweg_de_vries":5}
    if contour:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[8, 3])
        
        cf = ax1.contourf(time, env.domain_coordinates, env.state_traj, levels=50, 
                          extend="both", cmap=plt.get_cmap(contour_cmap_dict[env.id]))
        fig.colorbar(cf, orientation="vertical", label="state", format="%.2f")
        ax1.set_xlabel("simulation time")
        ax1.set_ylabel("domain coordinates")

        cmap2 = plt.get_cmap("jet")(np.linspace(0, 1, int(env.n_steps/sparcity_constant_dict[env.id])+1))
        for t in range(0, env.n_steps, sparcity_constant_dict[env.id]):
            ax2.plot(env.domain_coordinates, env.state_traj[:, t], color=cmap2[int(t/sparcity_constant_dict[env.id])], alpha=0.8)
        sm = plt.cm.ScalarMappable(
            cmap=plt.get_cmap("jet"), norm=plt.Normalize(time[0], time[-1])
        )
        plt.colorbar(sm, ax=ax2, label="simulation time")
        ax2.set_xlabel("domain coordinates")
        ax2.set_ylabel("state")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "contour.png"), dpi=dpi)
        if display:
            plt.show()
        plt.close(fig)

    if surface3d:
        fig = plt.figure(figsize=(7, 3.7))
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=30, azim=210)

        time_mesh, domain_coordinates_mesh = np.meshgrid(time, env.domain_coordinates)
        cf = ax.plot_surface(
            time_mesh, domain_coordinates_mesh, env.state_traj, 
            cmap=plt.get_cmap(contour_cmap_dict[env.id]), 
            rcount=env.n_state, ccount=env.n_steps,
        )
        ax.set_xlabel("simulation time")
        ax.set_ylabel("domain coordinates")
        ax.zaxis._axinfo['juggled'] = (1, 0, 2)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "surface3d.png"), dpi=dpi, bbox_inches='tight')
        if display:
            plt.show()
        plt.close(fig)

def _plot_coupled_pde(env, save_dir: str = None, display: bool = False, contour: bool = True, 
              surface3d: bool = True, dpi: int = 100):
    """Plot the state trajectory of a coupled PDE task (e.g., Wave and Schrodinger).
        Args:
            env: object, the tested linear control environment.
            save_dir: str, the path of the test folder.
            discrete_time: bool, whether to plot the state trajectory in discrete time.
            contour: bool, whether to plot and save the contour of the state trajectory.
            surface3d: bool, whether to plot and save the 3D surface of the state trajectory.
            dpi: int, the resolution of the saved figures.

        Returns:
            None.
    """
    # check whether save_dir is valid
    if not os.path.exists(save_dir):
        # handled by the helpers/data_processing/save
        raise FileNotFoundError(f"{save_dir} does not exist")
    if env.state_traj is None:
        raise ValueError("State_traj is empty. Please run the controller first.")

    time = np.linspace(0, env.n_steps * env.sample_time, env.n_steps + 1)
 
    contour_cmap_dict = {"schrodinger":["Spectral_r", "plasma"], "wave":["viridis", "plasma"]}
    sparcity_constant_dict = {"schrodinger":20, "wave":4}
    if contour:
        # plot two contour in one figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=[8, 6])
        
        cf1 = ax1.contourf(time, env.domain_coordinates, env.state_traj[: env.n_state_half, :], levels=50, 
                          extend="both", cmap=plt.get_cmap(contour_cmap_dict[env.id][0]))
        fig.colorbar(cf1, orientation="vertical", label="state (first half)", format="%.2f")
        ax1.set_xlabel("simulation time")
        ax1.set_ylabel("domain coordinates")

        cmap2 = plt.get_cmap("jet")(np.linspace(0, 1, int(env.n_steps/sparcity_constant_dict[env.id])+1))
        for t in range(0, env.n_steps, sparcity_constant_dict[env.id]):
            ax2.plot(env.domain_coordinates, env.state_traj[: env.n_state_half, t], color=cmap2[int(t/sparcity_constant_dict[env.id])], alpha=0.8)
            ax4.plot(env.domain_coordinates, env.state_traj[env.n_state_half :, t], color=cmap2[int(t/sparcity_constant_dict[env.id])], alpha=0.8)
        sm = plt.cm.ScalarMappable(
            cmap=plt.get_cmap("jet"), norm=plt.Normalize(time[0], time[-1])
        )
        plt.colorbar(sm, ax=ax2, label="simulation time")
        ax2.set_xlabel("domain coordinates")
        ax2.set_ylabel("state (first half)")

        cf3 = ax3.contourf(time, env.domain_coordinates, env.state_traj[env.n_state_half :, :], levels=50, 
                          extend="both", cmap=plt.get_cmap(contour_cmap_dict[env.id][1]))
        fig.colorbar(cf3, orientation="vertical", label="state (second half)", format="%.2f")
        ax3.set_xlabel("simulation time")
        ax3.set_ylabel("domain coordinates")

        plt.colorbar(sm, ax=ax4, label="simulation time")
        ax4.set_xlabel("domain coordinates")
        ax4.set_ylabel("state (second half)")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "contour.png"), dpi=dpi)
        if display:
            plt.show()
        plt.close(fig)

    if surface3d:
        fig = plt.figure(figsize=(7, 9))
        ax1 = fig.add_subplot(211, projection="3d")
        ax2 = fig.add_subplot(212, projection="3d")
        ax1.view_init(elev=30, azim=210)
        ax2.view_init(elev=30, azim=210)

        time_mesh, domain_coordinates_mesh = np.meshgrid(time, env.domain_coordinates)
        cf1 = ax1.plot_surface(
            time_mesh, domain_coordinates_mesh, env.state_traj[: env.n_state_half, :], 
            cmap=plt.get_cmap(contour_cmap_dict[env.id][0]), 
            rcount=env.n_state_half, ccount=env.n_steps,
        )
        cf2 = ax2.plot_surface(
            time_mesh, domain_coordinates_mesh, env.state_traj[env.n_state_half :, :], 
            cmap=plt.get_cmap(contour_cmap_dict[env.id][1]), 
            rcount=env.n_state_half, ccount=env.n_steps,
        )
        ax1.set_xlabel("simulation time", fontsize=13)
        ax1.set_ylabel("domain coordinates", fontsize=13)
        ax1.tick_params(axis='x', labelsize=13)
        ax1.tick_params(axis='y', labelsize=13)
        ax1.tick_params(axis='z', labelsize=13)
        ax1.zaxis._axinfo['juggled'] = (1, 0, 2)

        ax2.set_xlabel("simulation time", fontsize=13)
        ax2.set_ylabel("domain coordinates", fontsize=13)
        ax2.tick_params(axis='x', labelsize=13)
        ax2.tick_params(axis='y', labelsize=13)
        ax2.tick_params(axis='z', labelsize=13)
        ax2.zaxis._axinfo['juggled'] = (1, 0, 2)


        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "surface3d.png"), dpi=dpi, bbox_inches='tight')
        if display:
            plt.show()
        plt.close(fig)

def _plot_linear(env, save_dir: str = None, display: bool = False, dpi: int = 100):
    """Plot the state trajectory of a linear control task.
        Args:
            env: object, the tested linear control environment.
            save_dir: str, the path of the test folder.
            display: bool, whether to plot the images using plt.show().
            dpi: int, the resolution of the saved figures.

        Returns:
            None.
    """
    if not os.path.exists(save_dir):
        # handled by the helpers/data_processing/save
        raise FileNotFoundError(f"{save_dir} does not exist")
    if env.state_traj is None:
        raise ValueError("State_traj is empty. Please run the controller first.")
  
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    time = np.linspace(0, env.n_steps * env.sample_time, env.n_steps + 1)

    for i in range(env.n_state):
        ax.plot(time, env.state_traj[i])
    ax.set_xlabel("time", fontsize=13)
    ax.set_ylabel("state", fontsize=13)
    if env.n_state <= 6:
        ax.legend([fr"$s_{i}$" for i in range(env.n_state)], loc="upper right", fontsize=13)
    plt.tick_params(axis='both', which='major', labelsize=13)  # Set font size for major ticks
    plt.tick_params(axis='both', which='minor', labelsize=13)  # Set font size for minor ticks

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "state.png"), dpi=dpi)
    if display:
        plt.show()
    plt.close(fig)


def _plot_linearPDE_eigen(env, save_dir: str = None, display: bool = False, dpi: int = 100):
    """Plot the eigen spectrum of A matrix of a linear PDE envrionment.
        Args:
            env: object, the tested linear PDE environment.
            save_dir: str, the path of the test folder.
            display: bool, whether to plot the images using plt.show().
            dpi: int, the resolution of the saved figures.

        Returns:
            None.
    """
    if not env.id in ["convection_diffusion_reaction", "wave", "schrodinger"]:
        raise ValueError(f"{env.id} is not a linear PDE environment.")
    fig = plt.figure(figsize=[7, 4])
    ax = fig.add_subplot(111)
    circle1 = plt.Circle((0, 0), 1, fc="white", ec="black", alpha=0.5)

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.add_patch(circle1)
    X, Y = np.real(np.linalg.eig(env.A)[0]), np.imag(np.linalg.eig(env.A)[0])
    ax.axvline(x=0, color="k", alpha=0.3)
    ax.axhline(y=0, color="k", alpha=0.3)
    if hasattr(env, "eigen") and env.eigen is not None:
        ax.scatter(np.real(env.eigen), np.imag(env.eigen), color="tab:gray", marker="s", s=30, label="Analytical Eigenvalues")
    ax.scatter(X, Y, color="tab:orange", marker="*", s=14, label="Numerical Eigenvalues")
    ax.set_xlabel("Re", fontsize=13)
    ax.set_ylabel("Im", fontsize=13)
    ax.legend(fontsize=12, loc='upper left')
    plt.tick_params(axis='both', which='major', labelsize=13)  # Set font size for major ticks
    plt.tick_params(axis='both', which='minor', labelsize=13)  # Set font size for minor ticks

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "eigen_spectrum.png"), dpi=dpi)
    if display:
        plt.show()
    plt.close(fig)
