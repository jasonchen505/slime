import os
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
x = torch.tensor([rank + 1.0], device=f"cuda:{local_rank}")
dist.all_reduce(x, op=dist.ReduceOp.SUM)
torch.cuda.synchronize(local_rank)
expected = world * (world + 1) / 2
if abs(float(x.item()) - expected) > 1e-4:
    raise RuntimeError(f"rank {rank}: got {x.item()} expected {expected}")
if rank == 0:
    print(f"nccl_all_reduce_ok world={world} sum={x.item()}")
dist.destroy_process_group()
