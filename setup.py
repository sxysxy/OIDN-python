from setuptools import setup, Extension
import os
import sys 
import subprocess

def check_macos_architecture():
    p = subprocess.Popen(["file", sys.executable], stdout=subprocess.PIPE)
    out, err = p.communicate()
    out = out.decode()
    if "x86_64" in out or "x64" in out:
        return 'x64'
    elif "arm64" in out or "aarch64" in out:
        return 'aarch64'
    else:
        raise RuntimeError("Unknown architecture of this machine, candidate architectures are x64 and aarch64")

if sys.platform == 'darwin':
    if not ('x64' in sys.argv):
        arch = check_macos_architecture()
    else:
        arch = 'aarch64'
    print(f"Detected architecture is {arch}.")
    if arch == 'aarch64':
        data_files = []
    elif arch == 'x64':
        data_files = ["oidn/lib.macos.x64/libOpenImageDenoise_core.2.1.0.dylib", 
                    "oidn/lib.macos.x64/libOpenImageDenoise_device_cpu.2.1.0.dylib",
                    "oidn/lib.macos.x64/libOpenImageDenoise.2.1.0.dylib",
                    "oidn/lib.macos.x64/libtbb.12.10.dylib"]
    else:
        raise RuntimeError("")
    platform = 'Mac OS-X'
elif sys.platform == 'linux':
    data_files = ["oidn/lib.linux.x64/libOpenImageDenoise_core.so.2.1.0", 
                  "oidn/lib.linux.x64/libOpenImageDenoise_device_cpu.so.2.1.0",
                  "oidn/lib.linux.x64/libOpenImageDenoise.so.2.1.0",
                  "oidn/lib.linux.x64/libtbb.so.12.10"]
    platform = 'Linux'
elif sys.platform == 'win32':
    data_files = ["oidn/lib.win.x64/OpenImageDenoise_core.dll", 
                  "oidn/lib.win.x64/OpenImageDenoise_device_cpu.dll",
                  "oidn/lib.win.x64/OpenImageDenoise.dll",
                  "oidn/lib.win.x64/tbb12.dll"]
    platform = 'Windows'
    
with open("MANIFEST.in", "w") as f:
    f.write(f"include {' '.join(data_files)}")

setup( 
    name = 'oidn',
    version = '0.4',
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