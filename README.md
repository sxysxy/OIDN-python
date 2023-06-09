# OIDN-python
Python Binding of [Intel Open Image Denoise](https://github.com/OpenImageDenoise/oidn) Version 0.3 alpha (Based on OIDN 1.4.3)

Install using pip.(support macos_aarch64, win_amd64, linux_x64).

```
pip install oidn==0.3a0
```

# Features(version 0.3 alpha)

## C API wrapper

- Wrap original OIDN C APIs using ctypes. functions are stripped oidn prefix, macros are stripped OIDN_ prefix. For example oidnNewDevice -> oidn.NewDevice, OIDN_FORMAT_FLOAT3 -> oidn.FORMAT_FLOAT3. 

- Discard buffer APIs, use numpy array as buffers.

## Object-Oriented Interface

OOP style interface will be finished in version 1.0

# Example denoising image

Denoise image rendered by a monte carlo ray tracer. [code](./tests/DenoiseCornellBox/DenoiseCornellBox.py)

```python 
import sys 
sys.path.append("../..")
import oidn
from PIL import Image
import numpy as np
import os
here = os.path.dirname(__file__)

img = np.array(Image.open(f"{here}/CornellBoxNoisy.png"), dtype=np.float32) / 255.0
result = np.zeros_like(img, dtype=np.float32)

device = oidn.NewDevice()
oidn.CommitDevice(device)

filter = oidn.NewFilter(device, "RT")
oidn.SetSharedFilterImage(filter, "color", img, oidn.FORMAT_FLOAT3, img.shape[1], img.shape[0])
oidn.SetSharedFilterImage(filter, "output", result, oidn.FORMAT_FLOAT3, img.shape[1], img.shape[0])
oidn.CommitFilter(filter)
oidn.ExecuteFilter(filter)

result = np.array(np.clip(result * 255, 0, 255), dtype=np.uint8)
resultImage = Image.fromarray(result)
resultImage.save(f"{here}/CornellBoxDenoised.png")

oidn.ReleaseFilter(filter)
oidn.ReleaseDevice(device)
```

The image in left is before denoised, rendered by a Monte-Carlo PathTracer, spp=10, width=height=1000. The image in right is after denoised.

<div>
<div style="width:48%; display: inline-block"> 
<img src="tests/DenoiseCornellBox/CornellBoxNoisy.png">
</div>
<div style="width:48%; display: inline-block"> 
<img src="tests/DenoiseCornellBox/CornellBoxDenoisedAsExample.png">
</div>
</div>

# Update Log

- 0.3alpha : Warp nearly full APIs in oidn.h. (excluding buffer APIs, buffers are substituted numpy array), add function \_\_doc\_\_, to be comprehensively test.
- 0.2.1 : Support win_amd64 and manylinux1_x86_64 platform.
- 0.2 : Wrap basic device and filter APIs, Initial support for macosx_12_0_arm64 platform.

