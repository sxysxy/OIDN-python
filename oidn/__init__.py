import sys
import os
import ctypes
import subprocess
import oidn.capi as capi
from oidn.capi import *
import importlib
import torch
from PIL import Image

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
        self.__device_handle : int = NewDevice(d)
        #self.raise_if_error()
        CommitDevice(self.device_handle)
        #self.raise_if_error()
        self.type = device_type
        
    @property
    def error(self):    
        '''
        Returns a tuple[error_code, error_message], the same as oidn.GetDeviceError.
        '''
        return GetDeviceError(self.device_handle)
    
    def raise_if_error(self):
        '''
        Raise a RuntimeError if an error occured.
        '''
        err = self.error
        if not (err is None):
            if err[0] != 0:
                raise RuntimeError(err[1])
            
    def release(self):
        '''
        Call ReleaseDevice with self.device_handle
        '''
        if self.device_handle:  #not 0, not None
            ReleaseDevice(self.device_handle)
        self.native_handle = 0
        
    @property
    def is_cpu(self):
        '''
        Indicate whether it is a CPU device.
        '''
        return self.type == 'cpu'

    @property
    def is_cuda(self):
        '''
        Indicate wheter it is a CUDA device.
        '''
        return self.type == 'cuda'
    
    @property
    def device_handle(self):
        '''
        Returns the device handle
        '''
        return self.__device_handle
    
_torch = None
def _lazy_load_torch(raise_if_no_torch = True):  
    global _torch 
    if _torch is None:
        try:
            _torch = importlib.import_module("torch")
        except:
            if raise_if_no_torch:
                raise
            else:
                return None
        if not _torch.cuda.is_available():
            raise RuntimeError("Requires torch.cuda.is_available(), please check your torch installation.")
        return _torch
    else:
        return _torch
        
    
class Buffer(AutoReleaseByContextManaeger):
    
    def __init__(self, device : Device, width = 0, height = 0) -> None:
        '''
        Do not call this.
        '''
        self.device = device
        self.buffer_delegate = None
        self.format = None
        self.channel_first = False
        self.__width = width
        self.__height = height
        
    def release(self):
        '''
        Release corresponding resources.
        '''
        self.buffer_delegate = None
 
    @classmethod
    def create(cls, width : int, height : int, channels = 3, channel_first = False, device : Device = None, use_cupy = False, dtype=np.float32):
        r'''
        Create a buffer.
        
        Args:
            width    : width in pixel
            height   : height in pixel
            channels : channels of the image, it could be 0 or None.
            channel_first : If it is true and channels is not zero(None), self.buffer_delegate will be shaped to (channles, height, width), otherwise (height, width, channels). 
                            If the chennels parameter is zero(None), the shape will be (height, width) regardless channel_first.
            device   : Device. If is_cpu, self.buffer_delegate will be a numpy.ndarray, otherwise, if use_cupy is specified, the buffer_delegate will be a cupy.ndarray, otherwise it will be a torch.Tensor with device='cuda'.
            use_cupy : Use cupy, it is not implemented in OIDN-python 0.4.
            dtype    : could be np.float32, torch.float16(if supported)
        '''
        bf = cls(device, width, height)
        storage_shape = None
        if channels == 0 or channels is None:
            storage_shape = (height, width)
        else:
            if channel_first:
                storage_shape = (channels, height, width)
            else:
                storage_shape = (height, width, channels)
        
        if device.is_cpu:
            bf.buffer_delegate = np.zeros(shape=storage_shape, dtype=dtype)
        else:
            if use_cupy:
                raise NotImplementedError("Not implemented...")
            else:
                torch = _lazy_load_torch(raise_if_no_torch=False)
                if torch:
                    bf.buffer_delegate = torch.zeros(shape=storage_shape, dtype=dtype)
                else:
                    raise RuntimeError("torch is not installed")
        
        F32_FMTS = [FORMAT_FLOAT, FORMAT_FLOAT, FORMAT_FLOAT2, FORMAT_FLOAT3, FORMAT_FLOAT4]
        F16_FMTS = [FORMAT_HALF, FORMAT_HALF, FORMAT_HALF2, FORMAT_HALF3, FORMAT_HALF4]
        
        if channels is None:
            channels = 0
        if dtype == np.float32:
            bf.format = F32_FMTS[channels]
        else:
            torch = _lazy_load_torch(raise_if_no_torch=False)
            if torch and dtype == torch.float16:
                bf.format = F16_FMTS[channels]
            else:
                raise RuntimeError("torch is not installed")
        bf.channel_first = channel_first
        return bf    
    
    @classmethod
    def load(cls, device : Device, source, normalize, copy_data=True):
        '''
        Create a Buffer object from a data source.
        Args:
            device    : Device of the new Buffer object
            soruce    : Data source, could be PIL.Image, numpy.ndarray, torch.Tensor. If it is PIL.Image, copy_data will always be True.
            normalize : Normalize values into [0,1] by dividing 255(if source.dtype is uint8) or 65535(if source.dtype is uint16), useful for Image objects, if it is True, copy_data should also be True.  
            copy_data : Copy the source's data into a new container.
        '''
        if normalize and (not copy_data):
            raise RuntimeError("Setting div255 = True requires copy_data = True")
        
        if isinstance(source, Image.Image):
            #source : Image = source
            x_np = np.array(source)
            if normalize:
                if x_np.dtype == np.uint8:
                    x_np = x_np.astype(np.float32) / 255.0
                elif x_np.dtype == np.uint16:
                    x_np = x_np.astype(np.float32) / 65535.0
        elif isinstance(source, np.ndarray):
            if not copy_data:
                x_np = source
            else:
                x_np = np.array(source)
            if normalize:
                if x_np.dtype == np.uint8:
                    x_np = x_np.astype(np.float32) / 255.0
                elif x_np.dtype == np.uint16:
                    x_np = x_np.astype(np.float32) / 65535.0
        else:
            torch = _lazy_load_torch(False)
            if torch and isinstance(source, torch.Tensor):
                if source.is_cuda:
                    if not copy_data:
                        x_pt = source 
                    else:
                        x_pt = torch.tensor(source)
                    if normalize:
                        if x_pt.dtype == torch.uint8:
                            x_pt = x_pt.float() / 255.0
                        elif x_pt.dtype == torch.short:
                            x_pt = x_pt.float() / 65535.0 
                else:
                    raise RuntimeError("Requires source.is_cuda when source is a torch.Tensor")
            else:
                raise NotImplementedError(f"Not implemented sharing buffer from {type(source)}")

        
        if isinstance(source, Image.Image) or isinstance(source, np.ndarray):
            bf = Buffer.create(x_np.shape[1], x_np.shape[0], x_np.shape[2] if len(x_np.shape) > 2 else 0, 
                               channel_first=False, device=device, use_cupy=False)
            if device.is_cpu:
                bf.buffer_delegate = x_np
            else:
                bf.buffer_delegate = torch.tensor(x_np, device='cuda')
                
        else: # torch.Tensor, checked previously 
            bf = Buffer.create(x_pt.shape[1], x_pt.shape[0], x_pt.shape[2] if len(x_pt.shape) > 2 else 0, 
                               channel_first=False, device=device, use_cupy=False)
            
            if device.is_cpu:
                bf.buffer_delegate = x_pt.detach().cpu().numpy()
            else:
                bf.buffer_delegate = x_pt
        
        return bf
        
        
    def to_tensor(self):
        '''
        Returns:
            torch.Tensor
        '''
        if isinstance(self.buffer_delegate, torch.Tensor):
            return self.buffer_delegate
        else:
            raise RuntimeError("Can't convert cpu buffer to torch.Tensor")
    
    def to_array(self):
        '''
        Returns:
            numpy.ndarray
        '''
        if isinstance(self.buffer_delegate, np.ndarray):
            return self.buffer_delegate
        else:
            return self.buffer_delegate.detach().cpu().numpy()
    
    @property
    def width(self):
        '''
        Get width
        '''
        return self.__width
    
    @property
    def height(self):
        '''
        Get height
        '''
        return self.__height

class Filter(AutoReleaseByContextManaeger):
    def __init__(self, device : Device, type : str) -> None:
        r'''
        Args:
            device : oidn.Device
            type   : 'RT' or 'RTLightmap'
        '''
        self.device = device
        self.__filter_handle : int = NewFilter(device_handle=device.device_handle, type=type)
        if not self.filter_handle:
            raise RuntimeError("Can't create filter")
        self.device.raise_if_error()
        CommitFilter(self.__filter_handle)
        self.device.raise_if_error()
        
    @property
    def filter_handle(self) -> int:
        r'''
        Returns the handle of filter.
        '''
        return self.__filter_handle
    
    def release(self):
        r'''
        Call ReleaseFilter with self.fitler_handle
        '''
        if self.__filter_handle:
            ReleaseFilter(self.__filter_handle)
        self.__filter_handle = 0
        
    def set_image(self, name : str, buffer : Buffer):
        r'''
        Set image buffer for the filter.
        
        Args:
            name    : color/albedo/normal/output
            ------- 
                color : input beauty image (3 channels, LDR values in [0, 1] or HDR values in [0, +∞), values being interpreted such that, after scaling with the inputScale parameter, a value of 1 corresponds to a luminance level of 100 cd/m²)
                albedo (only support RT filter) : input auxiliary image containing the albedo per pixel (3 channels, values in [0, 1])
                normal (only support RT filter) : input auxiliary image containing the shading normal per pixel (3 channels, world-space or view-space vectors with arbitrary length, values in [-1, 1])
                output : output image (3 channels); can be one of the input images
            ------- 
            
            buffer  : Buffer object
        '''
        if self.device.is_cpu and (not buffer.device.is_cpu):
            raise RuntimeError("The filter is on CPU but the buffer is not")
        if self.device.is_cuda and (not buffer.device.is_cuda):
            raise RuntimeError("The filter is on CUDA but the buffer is not")
        
        def get_c_contiguous(b : Buffer):
            if isinstance(b.buffer_delegate, np.ndarray):
                return b.buffer_delegate.flags.c_contiguous
            else:
                return b.is_contiguous()
                
        def get_shape(b : Buffer):
            return b.buffer_delegate.shape
        
        def get_array_interface(b : Buffer):
            if isinstance(b.buffer_delegate, np.ndarray):
                return b.buffer_delegate.__array_interface__
            else:
                return b.buffer_delegate.__cuda_array_interface__
        
        SetSharedFilterImageEx(self.filter_handle, name, buffer,
                               get_shape=get_shape, check_c_contiguous=get_c_contiguous, get_array_interface=get_array_interface,
                               format=buffer.format, width=buffer.width, height=buffer.height)
        CommitFilter(self.filter_handle)
        self.device.raise_if_error()

    def execute(self):
        r'''
        Run the filter, wait until finished.
        '''
        if self.filter_handle:
            ExecuteFilter(self.filter_handle)
            self.device.raise_if_error()
        else:
            raise RuntimeError("Invalid filter handle")

    

    

    
