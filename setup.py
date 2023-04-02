from setuptools import setup
import sys 
if sys.platform == 'darwin':
    data_files = ["oidn/lib.macos.aarch64/libOpenImageDenoise.1.4.3.dylib"]
elif sys.platform == 'linux':
    data_files = ["oidn/lib.linux.x64/libOpenImageDenoise.so.1.4.3", "oidn/lib.linux.x64/libtbb.so.12.5"]
elif sys.platform == 'win32':
    data_files = ["oidn/lib.win.x64/OpenImageDenoise.dll", "oidn/lib.win.x64/tbb12.dll"]

setup( 
    name = 'oidn',
    version = '0.1',
    author = 'HfCloud',
    author_email = 'sxysxygm@gmail.com',
    description = 'A simple python binding for Intel OIDN',
    url="https://github.com/sxysxy/OIDN-python",
    package_dir='oidn',
    install_requirs=[
        'numpy >= 1.12.0',
        'pillow'
    ],
    data_files=data_files
)