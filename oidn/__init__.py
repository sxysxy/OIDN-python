import sys
import os
import ctypes
import oidn.capi as capi
from oidn.capi import *

oidn_version = '1.4.3'
oidn_py_version = '0.2.1'

def __load_lib():
    cur_path = os.path.dirname(__file__)
    if sys.platform == 'linux':
        return ctypes.CDLL(os.path.join(cur_path, f"lib.linux.x64/libOpenImageDenoise.so.{oidn_version}"))
    elif sys.platform == 'darwin':
        return ctypes.CDLL(os.path.join(cur_path, f"lib.macos.aarch64/libOpenImageDenoise.{oidn_version}.dylib"))
    elif sys.platform == 'win32':
        return ctypes.CDLL(os.path.join(cur_path, f"lib.win.x64/OpenImageDenoise.dll"))
    else:
        raise RuntimeError("Unsupported platform")
__lib_oidn = __load_lib()

capi.__init_by_lib(__lib_oidn)

    

    

    

    
