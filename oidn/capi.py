import ctypes
import os
from oidn.constants import *
import typing
import numpy as np

class RawFunctions:
    
    # Device 
    oidnNewDevice = None
    oidnCommitDevice = None
    oidnGetDeviceError = None
    oidnReleaseDevice = None
    
    # Filter API
    oidnNewFilter = None
    oidnSetSharedFilterImage = None
    oidnCommitFilter = None
    oidnExecuteFilter = None
    oidnReleaseFilter = None
    

def __init_by_lib(lib : ctypes.CDLL):
    type_map = {
        'i' : ctypes.c_int32,
        'l' : ctypes.c_long,
        'L' : ctypes.c_longlong,
        'z' : ctypes.c_size_t,
        'f' : ctypes.c_float,
        'd' : ctypes.c_double,
        'D' : ctypes.c_longdouble,
        'c' : ctypes.c_char,
        's' : ctypes.c_char_p,
        'p' : ctypes.c_void_p,
        'n' : None
    }
    def get_func(name, argtype, restype):
        f = lib.__getattr__(name)
        f.restype = type_map[restype]
        f.argtypes = [type_map[t] for t in argtype]
        return f
    
    RawFunctions.oidnNewDevice = get_func("oidnNewDevice", 'i', 'p')
    RawFunctions.oidnCommitDevice = get_func("oidnCommitDevice", 'p', 'n')
    RawFunctions.oidnGetDeviceError = get_func("oidnGetDeviceError", 'pp', 'i')
    RawFunctions.oidnReleaseDevice = get_func("oidnReleaseDevice", 'p', 'n')
    
    RawFunctions.oidnNewFilter = get_func("oidnNewFilter", 'ps', 'p')
    RawFunctions.oidnSetSharedFilterImage = get_func("oidnSetSharedFilterImage", 'pspizzzzz', 'n')
    RawFunctions.oidnCommitFilter = get_func("oidnCommitFilter", 'p', 'n')
    RawFunctions.oidnExecuteFilter = get_func("oidnExecuteFilter", 'p', 'n')
    RawFunctions.oidnReleaseFilter = get_func("oidnReleaseFilter", 'p', 'n')
    
def NewDevice(device_type : int = DEVICE_TYPE_DEFAULT) -> int:
    '''
    Create a new OIDN device
    
    Args:
        device_type(int) : OIDN_DEVICE_TYPE_XXX
    '''
    return RawFunctions.oidnNewDevice(device_type)

def CommitDevice(device_handle : int):
    RawFunctions.oidnCommitDevice(device_handle)

def GetDeviceError(device_handle : int):
    err = RawFunctions.oidnGetDeviceError(device_handle, 0)
    return err 

def ReleaseDevice(device_handle : int):
    RawFunctions.oidnReleaseDevice(device_handle)
    
def NewFilter(device_handle : int, type : str) -> int:
    '''
    Creates a new filter of the specified type (e.g. "RT")
    
    Args:
        device_handle(int) : Created by NewDevice
        type(str) : e.g. "RT“
    '''
    return RawFunctions.oidnNewFilter(device_handle, bytes(type, 'ascii'))

def SetSharedFilterImage(filter_handle : int, name : str, data : np.ndarray, format : int, width : int, height : int, byteOffset : int = 0, bytePixelStride : int = 0, byteRowStride : int = 0):
    """
    Set filter parameters
    
    Args:
        filter_handle(int): Created by NewFilter 
        name(str): color/albedo/normal/output, See document of OIDN.
        data(np.array): data buffer, should be correct in size and dtype; should be c_contiguous when name == 'output'
        format:  Should be oidn.FORMAT_FLOAT3 for image
        width: width in pixel.
        height: height in pixel.
        byteOffset: default to 0
        bytePixel: default to 0
        byteRawStride: default to 0
    """
    desired_dim3 = [0, 1, 2, 3, 4]
    desired_data_shape = (height, width, desired_dim3[format])
    if not data.shape == desired_data_shape:
        raise RuntimeError(f"The shape of the data should be {desired_data_shape}")

    if name == "output" and not data.flags.c_contiguous:
        raise RuntimeError(f"When name == output, the data should be c_contiguous")
    
    if not data.flags.c_contiguous:
        data = np.ascontiguousarray(data)
    RawFunctions.oidnSetSharedFilterImage(filter_handle, bytes(name, 'ascii'), data.__array_interface__['data'][0], format, width, height, byteOffset, bytePixelStride, byteRowStride)
    
def CommitFilter(filter_handle : int):
    RawFunctions.oidnCommitFilter(filter_handle)
    
def ExecuteFilter(filter_handle : int):
    RawFunctions.oidnExecuteFilter(filter_handle)
    
def ReleaseFilter(filter_handle : int):
    RawFunctions.oidnReleaseFilter(filter_handle)

    

    