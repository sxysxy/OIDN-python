from setuptools import setup, Extension
import os
import sys 
if sys.platform == 'darwin':
    data_files = ["oidn/lib.macos.aarch64/libOpenImageDenoise.1.4.3.dylib"]
    platform = 'Mac OS-X'
elif sys.platform == 'linux':
    data_files = ["oidn/lib.linux.x64/libOpenImageDenoise.so.1.4.3", "oidn/lib.linux.x64/libtbb.so.12.5"]
    platform = 'Linux'
elif sys.platform == 'win32':
    data_files = ["oidn/lib.win.x64/OpenImageDenoise.dll", "oidn/lib.win.x64/tbb12.dll"]
    platform = 'Windows'

# data_files = ["oidn/lib.macos.aarch64/libOpenImageDenoise.1.4.3.dylib", 
#              "oidn/lib.linux.x64/libOpenImageDenoise.so.1.4.3", "oidn/lib.linux.x64/libtbb.so.12.5", 
#              "oidn/lib.win.x64/OpenImageDenoise.dll", "oidn/lib.win.x64/tbb12.dll"]
    
with open("MANIFEST.in", "w") as f:
    f.write(f"include {' '.join(data_files)}")

setup( 
    name = 'oidn',
    version = '0.2.1',
    author = 'HfCloud',
    author_email = 'sxysxygm@gmail.com',
    description = 'A simple python binding for Intel OIDN',
    license='Apache-2.0',
    url="https://github.com/sxysxy/OIDN-python",
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3',
    ],
    packages=['oidn'],
    package_data={
        'oidn' : data_files
    },
  #  ext_modules=[Extension(data_files[0],[], optional=True, runtime_library_dirs=[os.path.dirname(data_files[0])])],
    include_package_data=True,
    python_requires='>=3',
    install_requires=[
        'numpy >= 1.12.0',
        'pillow'
    ],
    platforms=[platform]
)