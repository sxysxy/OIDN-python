# OIDN-python command line tool

import argparse
from PIL import Image 
import numpy as np
import oidn

if __name__ == "__main__":
    with oidn.Device('cpu') as device, oidn.Filter(device, 'RT') as filter:
        pass
    