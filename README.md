# OIDN-python
Python Binding of [Intel Open Image Denoise](https://github.com/OpenImageDenoise/oidn) Version 0.2 (Based on OIDN 1.4.3)

Install using pip(Only available for Apple Silicon macos for 0.2 version)

```
pip install oidn
```

# Features(version 0.2)

## C API wrapper

Wrap OIDN C API, functions are stripped oidn prefix, macros are stripped OIDN_ prefix. For example oidnNewDevice -> oidn.NewDevice, OIDN_FORMAT_FLOAT3 -> oidn.FORMAT_FLOAT3.

### Example denoising image

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

## Object-Oriented Interface

OOP style interface will be finished in version 1.0.

