import sys
import os
import ctypes
import oidn.capi as capi
from oidn.capi import *

oidn_version = '1.4.3'
oidn_py_version = '0.1'

def __load_lib():
    cur_path = os.path.dirname(__file__)
    if sys.platform == 'linux':
        raise NotImplementedError("Not implemented for linux")
    elif sys.platform == 'darwin':
        return ctypes.CDLL(os.path.join(cur_path, f"lib.macos.aarch64/libOpenImageDenoise.{oidn_version}.dylib"))
    elif sys.platform == 'win32':
        raise NotImplementedError("Not implemented for win32")
    else:
        raise RuntimeError("Unsupported platform")
__lib_oidn = __load_lib()

capi.__init_by_lib(__lib_oidn)

    

    

    

    
