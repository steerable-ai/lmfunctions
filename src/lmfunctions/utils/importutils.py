import subprocess
from importlib import import_module
from types import ModuleType


def pip_install(packages, flags=[]):

    for package in packages:
        command_list = ["pip", "install", package] + flags
        if (
            input(
                f"Package '{package}' is needed. Do you want to run "
                f"``{' '.join(command_list)}''? (y/n)"
            ).lower()
            == "y"
        ):
            subprocess.run(command_list)
            print(f"Package '{package}' installed successfully.")
        else:
            return False
    return True


def install_callback(name, package, **kwargs):
    if pip_install([name], **kwargs):
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
