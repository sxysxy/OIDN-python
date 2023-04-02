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