from .files import download_url, files_exist, makedirs
from .feeding_structs import InputParameter, ModuleParameter, HyperParameter, ModelInput, GraphInput
from .initialization import glorot, zeros, ones
from .transformation import *

device_data = [None]


def get_device():
    return device_data[0]


def set_device(device):
    device_data[0] = device


def debug(*args, **kwargs):
    kwargs['flush'] = kwargs.get('flush', True)
    print(*args, **kwargs)
