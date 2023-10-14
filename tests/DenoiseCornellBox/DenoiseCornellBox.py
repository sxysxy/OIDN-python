from pathlib import Path
import sys
import numpy as np
from PIL import Image

here = Path(__file__).parent.absolute()
sys.path.append(here.parent.parent.absolute().as_posix())

import oidn

img = np.array(Image.open((here / "CornellBoxNoisy.png").as_posix()), dtype=np.float32) / 255.0
result = np.zeros_like(img, dtype=np.float32)

device = oidn.NewDevice()
oidn.CommitDevice(device)

filter = oidn.NewFilter(device, "RT")
oidn.SetSharedFilterImage(
    filter, "color", img, oidn.FORMAT_FLOAT3, img.shape[1], img.shape[0]
)
oidn.SetSharedFilterImage(
    filter, "output", result, oidn.FORMAT_FLOAT3, img.shape[1], img.shape[0]
)
oidn.CommitFilter(filter)
oidn.ExecuteFilter(filter)

result = np.array(np.clip(result * 255, 0, 255), dtype=np.uint8)
resultImage = Image.fromarray(result)
resultImage.save(f"{here}/CornellBoxDenoised.png")

oidn.ReleaseFilter(filter)
oidn.ReleaseDevice(device)
