import oidn
import inspect
import re

with open("APIs.md", "w") as f:
    f.write("# C APIs\n")
    
    f.write("## Free functions:\n")
    capi_set = [
        "NewDevice", "CommitDevice", "GetDeviceError", "ReleaseDevice", "RetainDevice",
        "SetDeviceBool", "SetDevice1b", "SetDeviceInt", "SetDevice1i", "GetDeviceInt", "GetDevice1i", "GetDeviceBool", "GetDevice1b",
        "NewFilter", "SetSharedFilterImage", "UnsetFilterImage", "RemoveFilterImage", 
        "SetSharedFilterData", "UpdateFilterData", "UnsetFilterData", "RemoveFilterData",
        "GetFilterInt", "GetFilter1i", "GetFilterBool", "GetFilter1b", "GetFilterFloat", "GetFilter1f",
        "SetFilterBool", "SetFilter1b", "SetFilterInt", "SetFilter1i", "SetFilterFloat", "SetFilter1f",
        "CommitFilter", "ExecuteFilter", "ReleaseFilter", "RetainFilter"
    ]
    
    for name in capi_set:
        obj = getattr(oidn, name)
        f.write(f"### {name}{inspect.signature(obj)}:\n```\n{obj.__doc__.strip()}\n```\n")
    
    f.write("## Class RawFunction\n")
    f.write("These functions are FFI objects for corresponding native functions in the OIDN dynamic linked library.\n")
    for name, obj in inspect.getmembers(oidn.RawFunctions):
        if name.startswith("oidn"):
            f.write(f"### {name}\n")
    
    f.write("# Pythonic APIs\n")
    
    pyapi_classes = [
        "Device", "Filter", "Buffer"
    ]
    for name in pyapi_classes:
        obj = getattr(oidn, name)
        f.write(f"## Class {name}\n")
        for method_name, method_obj in inspect.getmembers(obj):
            if method_name.startswith("__") and method_name != "__init__":
                continue
            method_name = re.sub("_", "\\_", method_name)
            doc = method_obj.__doc__
            if not doc or len(doc) == 0:
                doc = "(No document)"
            doc = doc.strip()
            if inspect.isroutine(method_obj):
                f.write(f"### {method_name}{inspect.signature(method_obj)}: <div style=\"text-align: right; float: right; color: #215f11\">method</div> \n```\n{doc}\n```\n")
            elif isinstance(method_obj, property):
                f.write(f"### {method_name} <div style=\"text-align: right; float: right; color: #21138d\">property</div>  \n```\n{doc}\n```\n")