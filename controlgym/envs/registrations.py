from controlgym.envs import (LinearControlEnv, AllenCahnEnv, BurgersEnv, CahnHilliardEnv, 
                      ConvectionDiffusionReactionEnv, FisherEnv, GinzburgLandauEnv,
                      KortewegDeVriesEnv, KuramotoSivashinskyEnv, SchrodingerEnv, WaveEnv)


def make(id: str, **kwargs):
    """Constructor of environment object such that when the id of the environment is provided,
        along with other optional arguments for this environmenet, then make() returns an instance
        of the environment.

        Args:
            id (str): the id of the environment.
            **kwargs: other optional arguments for the environment.

        Returns:
            An instance of the environment.
    """
    linear_control_id_list = ["toy", "ac1", "ac2", "ac3", "ac4", "ac5", "ac6", 
                              "ac7", "ac8", "ac9", "ac10", "bdt1", "bdt2", "cbm", 
                              "cdp", "cm1", "cm2", "cm3", "cm4", "cm5", "dis1", 
                              "dis2", "dlr", "he1", "he2", "he3", "he4", "he5", "he6", 
                              "iss", "je1", "je2", "lah", "pas", "psm", "rea", "umv"]
    pde_id_list = ["allen_cahn", "burgers", "cahn_hilliard", "convection_diffusion_reaction", 
                   "fisher", "ginzburg_landau", "korteweg_de_vries", "kuramoto_sivashinsky", 
                   "schrodinger", "wave"]
    if id in linear_control_id_list:
        return LinearControlEnv(id=id, **kwargs)
    elif id in pde_id_list:
        return _call_function_by_name(_to_camel_env_id(id), **kwargs)
    else:
        raise ValueError("Invalid environment id. " + "Valid environment ids include: ", linear_control_id_list+pde_id_list)

def _to_camel_env_id(id: str):
    """Change the string id of the environment to camel case.

        Args:
            id (str): the id of the environment.

        Returns:
            The camel case id of the environment.
    """
    if len(id) == 0:
        return id
    else:
        camel_id = (id.replace("-", " ").replace("_", " ")).split()
        return camel_id[0].capitalize() + ''.join(i.capitalize() for i in camel_id[1:]) + "Env"

def _call_function_by_name(name: str, **kwargs):
    """Call the constructor of the PDE environment by its name.

        Args:
            id (str): the id of the environment.
            **kwargs: other optional arguments for the environment.

        Returns:
            An instance of the environment.
    """
    if name in globals() and callable(globals()[name]):
        # Check if the function exists in the global scope and is callable
        func = globals()[name]
        return func(**kwargs)
    else:
        # handled by the check statement in make()
        return None  # Function not found