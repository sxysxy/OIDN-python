import sys
import os
import ctypes
import oidn.capi as capi
from oidn.capi import *

oidn_version = '2.1.0'
oidn_py_version = '0.3.1alpha'

def __load_lib():
    cur_path = os.path.dirname(__file__)
    if sys.platform == 'linux':
        ctypes.CDLL(os.path.join(cur_path, f"lib.linux.x64/libOpenImageDenoise_core.so.{oidn_version}"))
        ctypes.CDLL(os.path.join(cur_path, f"lib.linux.x64/libOpenImageDenoise_device_cpu.so.{oidn_version}"))
        return ctypes.CDLL(os.path.join(cur_path, f"lib.linux.x64/libOpenImageDenoise.so.{oidn_version}"))
    elif sys.platform == 'darwin':
        ctypes.CDLL(os.path.join(cur_path, f"lib.macos.aarch64/libOpenImageDenoise_core.{oidn_version}.dylib"))
        ctypes.CDLL(os.path.join(cur_path, f"lib.macos.aarch64/OpenImageDenoise_device_cpu.{oidn_version}.dylib"))
        return ctypes.CDLL(os.path.join(cur_path, f"lib.macos.aarch64/libOpenImageDenoise.{oidn_version}.dylib"))
    elif sys.platform == 'win32':
        ctypes.CDLL(os.path.join(cur_path, f"lib.win.x64/OpenImageDenoise_core.dll"))
        ctypes.CDLL(os.path.join(cur_path, f"lib.win.x64/OpenImageDenoise_device_cpu.dll"))
        return ctypes.CDLL(os.path.join(cur_path, f"lib.win.x64/OpenImageDenoise.dll"))
    else:
        raise RuntimeError("Unsupported platform")
__lib_oidn = __load_lib()

capi.__init_by_lib(__lib_oidn)

    

    

    

    
