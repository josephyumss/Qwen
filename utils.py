import torch

def fmt(b):  # bytes → human-readable
    for unit in ["B","KB","MB","GB","TB"]:
        if b < 1024: return f"{b:.2f} {unit}"
        b /= 1024
    return f"{b:.2f} PB"

def print_cuda_mem(dev=None, note=""):
    if not torch.cuda.is_available():
        print("[CUDA OFF] running on CPU")
        return
    if dev is None:
        dev = torch.cuda.current_device()
    torch.cuda.synchronize(dev)
    allocated = torch.cuda.memory_allocated(dev)       
    reserved  = torch.cuda.memory_reserved(dev)      
    max_alloc = torch.cuda.max_memory_allocated(dev)  
    max_res   = torch.cuda.max_memory_reserved(dev)
    total = torch.cuda.get_device_properties(dev).total_memory
    free, total_nv = torch.cuda.mem_get_info()     

    name = torch.cuda.get_device_name(dev)
    print(f"[{note}] GPU{dev} {name}")
    print(f"  allocated: {fmt(allocated)}   | max_alloc: {fmt(max_alloc)}")
    print(f"  reserved : {fmt(reserved)}    | max_res  : {fmt(max_res)}")
    print(f"  total    : {fmt(total)}       | driver_free: {fmt(free)}")