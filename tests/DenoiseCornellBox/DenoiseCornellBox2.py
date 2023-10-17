from pathlib import Path
import sys
import numpy as np
from PIL import Image

here = Path(__file__).parent.absolute()
sys.path.append(here.parent.parent.absolute().as_posix())

import oidn

with oidn.Device('cpu') as device, oidn.Filter(device, 'RT') as filter:
    input = oidn.Buffer.load(device, Image.open((here / "CornellBoxNoisy.png").as_posix()), div255=True)
    output = oidn.Buffer.create(input.width, input.height, device=device)
    filter.set_image("color", input)
    filter.set_image("output", output)
    filter.execute()
    Image.fromarray( np.array(np.clip(output.to_array() * 255, 0, 255), dtype=np.uint8) ).save(f"{here}/CornellBoxDenoised.png")