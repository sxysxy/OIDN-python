DEVICE_TYPE_DEFAULT = 0
'''
Select device automatically
'''

DEVICE_TYPE_CPU = 1
'''
CPU device
'''
    
FORMAT_UNDEFINED = 0,
FORMAT_FLOAT    = 1
FORMAT_FLOAT2   = 2
FORMAT_FLOAT3   = 3
FORMAT_FLOAT4   = 4
# FORMAT_HALF     = 257
# FORMAT_HALF2    = 258
# FORMAT_HALF3    = 259
# FORMAT_HALF4    = 260
    
ACCESS_READ      = 0
'''
Read-only access
'''

ACCESS_WRITE     = 1
'''
Write-only access
'''

ACCESS_READ_WRITE = 2
'''
Read and write access
'''

ACCESS_WRITE_DISCARD = 3
'''
Write-only access, previous contents discarded
'''


ERROR_NONE                 = 0
ERROR_UNKNOWN              = 1
ERROR_INVALID_ARGUMENT     = 2
ERROR_INVALID_OPERATION    = 3
ERROR_OUT_OF_MEMORY        = 4
ERROR_UNSUPPORTED_HARDWARE = 5
ERROR_CANCELLED            = 6