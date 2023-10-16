import sys
import os
import ctypes
import subprocess
import oidn.capi as capi
from oidn.capi import *

oidn_version = '2.1.0'
oidn_py_version = '0.4'

def __load_lib():
     
    cur_path = os.path.dirname(__file__)
    if sys.platform == 'linux':
        ctypes.CDLL(os.path.join(cur_path, f"lib.linux.x64/libOpenImageDenoise_core.so.{oidn_version}"))
        ctypes.CDLL(os.path.join(cur_path, f"lib.linux.x64/libOpenImageDenoise_device_cpu.so.{oidn_version}"))
        return ctypes.CDLL(os.path.join(cur_path, f"lib.linux.x64/libOpenImageDenoise.so.{oidn_version}"))
    elif sys.platform == 'darwin':
        p = subprocess.Popen(["file", sys.executable], stdout=subprocess.PIPE)
        out, err = p.communicate()
        out = out.decode()
        if "x86_64" in out or "x64" in out:
            arch = 'x64'
        elif "arm64" in out or "aarch64" in out:
            arch = 'aarch64'
        else:
            raise RuntimeError("Unknown architecture of this machine, candidate architectures are x64 and aarch64")
        if arch == 'aarch64':
            ctypes.CDLL(os.path.join(cur_path, f"lib.macos.aarch64/libOpenImageDenoise_core.{oidn_version}.dylib"))
            ctypes.CDLL(os.path.join(cur_path, f"lib.macos.aarch64/libOpenImageDenoise_device_cpu.{oidn_version}.dylib"))
            return ctypes.CDLL(os.path.join(cur_path, f"lib.macos.aarch64/libOpenImageDenoise.{oidn_version}.dylib"))
        else:  #x64
            ctypes.CDLL(os.path.join(cur_path, f"lib.macos.x64/libOpenImageDenoise_core.{oidn_version}.dylib"))
            ctypes.CDLL(os.path.join(cur_path, f"lib.macos.x64/libOpenImageDenoise_device_cpu.{oidn_version}.dylib"))
            return ctypes.CDLL(os.path.join(cur_path, f"lib.macos.x64/libOpenImageDenoise.{oidn_version}.dylib"))
    elif sys.platform == 'win32':
        ctypes.CDLL(os.path.join(cur_path, f"lib.win.x64/OpenImageDenoise_core.dll"))
        ctypes.CDLL(os.path.join(cur_path, f"lib.win.x64/OpenImageDenoise_device_cpu.dll"))
        return ctypes.CDLL(os.path.join(cur_path, f"lib.win.x64/OpenImageDenoise.dll"))
    else:
        raise RuntimeError("Unsupported platform")
__lib_oidn = __load_lib()

capi.__init_by_lib(__lib_oidn)

class AutoReleaseByContextManaeger:
    def __enter__(self):
        return self
    
    def __exit__(self, _1, _2, _3):
        self.release()
        

class Device(AutoReleaseByContextManaeger):
    def __init__(self, device_type = 'cpu') -> None:
        r'''
        Create an OIDN device.
        
        Args:
            device_type: 'cpu' or 'cuda'
        '''
        if device_type == 'cpu':
            d = DEVICE_TYPE_CPU
        elif device_type == 'cuda':
            d = DEVICE_TYPE_CUDA
        else:
            raise RuntimeError("Requires device_type in ['cpu', 'cuda']")
        self.device_handle : int = NewDevice(d)
        CommitDevice(self.device_handle)
        self.type = type
        
    @property
    def error(self):    
        return GetDeviceError(self.device_handle)
    
    def raise_if_error(self):
        err = self.error
        if err is None:
            if err[0] != 0:
                raise RuntimeError(err[1])
            
    def release(self):
        if self.device_handle:  #not 0, not None
            ReleaseDevice(self.device_handle)
        self.native_handle = 0
        
    @property
    def is_cpu(self):
        return self.type == 'cpu'

    @property
    def is_cuda(self):
        return self.type == 'cuda'
        
    
class Buffer(AutoReleaseByContextManaeger):
    def __init__(self, device : Device) -> None:
        self.device = device
        self.buffer_delegate = None
        
    
    def release(self):
        self.buffer_delegate = None
    
    def to_tensor(self):
        '''
        Returns:
            torch.Tensor
        '''
        pass 
    
    def to_array(self):
        '''
        Returns:
            numpy.ndarray or cupy.ndarray
        '''
        pass

    @classmethod
    def shared_from(cls, source):
        pass
    

class Filter(AutoReleaseByContextManaeger):
    def __init__(self, device : Device, type : str) -> None:
        r'''
        Args:
            type: 'RT' or 'RTLightmap'
        '''
        self.device = device
        self.filter_handle : int = NewFilter(device_handle=device.device_handle, type=type)
        if not self.filter_handle:
            raise RuntimeError("Can't create filter")
        device.raise_if_error()
        CommitFilter(self.filter_handle)
        device.raise_if_error()
    
    def release(self):
        if self.filter_handle:
            ReleaseFilter(self.filter_handle)
        self.filter_handle = 0

    def execute(self) -> Buffer:
        pass

    

    

    
