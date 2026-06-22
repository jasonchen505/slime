import json
import torch

print("torch_version", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
count = torch.cuda.device_count()
print("device_count", count)
assert torch.cuda.is_available(), "CUDA is not available"
assert count >= 8, f"expected at least 8 GPUs, got {count}"
results = []
for i in range(8):
    torch.cuda.set_device(i)
    props = torch.cuda.get_device_properties(i)
    a = torch.randn((1024, 1024), device=f"cuda:{i}", dtype=torch.float16)
    b = torch.randn((1024, 1024), device=f"cuda:{i}", dtype=torch.float16)
    c = a @ b
    torch.cuda.synchronize(i)
    results.append({
        "gpu": i,
        "name": props.name,
        "total_mem_gb": round(props.total_memory / 1024**3, 2),
        "mean": float(c.float().mean().cpu()),
        "allocated_mb": round(torch.cuda.memory_allocated(i) / 1024**2, 2),
    })
print("single_process_results", json.dumps(results, ensure_ascii=False))
