import subprocess
from importlib import import_module
from types import ModuleType


def pip_install(packages):
    for package in packages:
        response = input(
            f"Package '{package}' is needed. Do you want to run "
            f"``pip install {package}''? (y/n)"
        )
        if response.lower() == "y":
            subprocess.run(["pip", "install", package])
            print(f"Package '{package}' installed successfully.")
        else:
            return False
    return True


def lazy_import(name, package=None, install_packages=None) -> ModuleType | None:
    """
    Lazily imports a module and handles the case when the module is not found.

    Args:
        module_name (str): The name of the module to import..

    Returns:
        ModuleType | None: The imported module if found, otherwise None.

    Raises:
        None
    """

    try:
        return import_module(name, package=package)
    except ImportError:
        install_packages = install_packages or [name]
        if pip_install(install_packages):
            return import_module(name, package=package)
        raise ImportError(f"The following packages are required: '{install_packages}'")
