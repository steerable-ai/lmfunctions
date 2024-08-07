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


def install_callback(name, package):
    if pip_install([name]):
        return import_module(name, package=package)
    else:
        return None


def lazy_import(
    name, package=None, import_error_callback=install_callback
) -> ModuleType | None:
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
    except ImportError as e:
        if import_error_callback:
            module = import_error_callback(name, package)
            if module:
                return module
        raise e
