"""
A use case of distributed traning of torch
"""
from concurrent.futures import process
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

""" Blocking p2p communication: 
Both processes stop until the communication is completed
"""
def run_p2p_block(rank, size):
    # Two processes must be placed in different devices, otherwise:
    # NCCL WARN Duplicate GPU detected : rank 0 and rank 1 both on CUDA device 1f000,
    # NCCL error This usually reflects invalid usage of NCCL library (such as too many async ops, too many collectives at once, mixing streams in a group, etc).
    torch.cuda.set_device(rank)
    tensor = torch.ones(1).cuda()
    if rank == 0:
        tensor += 1
        # Send tensor to process 1
        dist.send(tensor=tensor, dst=1)
        print("Hi, I have send tensor.")
    else:
        # Receive tensor form process 0
        dist.recv(tensor=tensor, src=0)
        print("Okey, I recevived.")
    print('Rank', rank, 'has data', tensor.item())

""" Non-blocking p2p communication."""
def run_p2p_noblock(rank, size):
    # Two processes must be placed in different devices, otherwise:
    # NCCL WARN Duplicate GPU detected : rank 0 and rank 1 both on CUDA device 1f000,
    # NCCL error This usually reflects invalid usage of NCCL library (such as too many async ops, too many collectives at once, mixing streams in a group, etc).
    torch.cuda.set_device(rank)
    tensor = torch.ones(1).cuda()
    req = None
    if rank == 0:
        tensor += 1
        # Send tensor to process 1
        req = dist.isend(tensor=tensor, dst=1)
        print("Hi, I have send tensor.")
    else:
        # Receive tensor form process 0
        req = dist.irecv(tensor=tensor, src=0)
        print("Okey, I recevived.")
    req.wait()
    print('Rank', rank, 'has data', tensor.item())

""" Collective Communication Example: All Reduce """
def run_all_reduce(rank, size):
    torch.cuda.set_device(rank)
    if rank == 0:
        tensor = torch.full([3,3], 1).cuda()
    if rank == 1:
        tensor = torch.full([3,3], 2).cuda()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print('Rank ', rank, ' has data ', tensor.tolist())

def run_reduce(rank, size):
    torch.cuda.set_device(rank)
    if rank == 0:
        tensor = torch.full([3,3], 1).cuda()
    if rank == 1:
        tensor = torch.full([3,3], 2).cuda()
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    print('Rank ', rank, ' has data ', tensor.tolist())    


def init_process(rank, size, fn, backend='nccl'):
    ##### Distributed Training Code
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    size = 2
    process = []
    # Three kinds of start mothod -> ['spawn', 'fork', 'fork server'], read more:
    # https://docs.python.org/3.10/library/multiprocessing.html#contexts-and-start-methods
    mp.set_start_method("spawn")
    for rank in range(size):
        # Multiprocess Code
        p = mp.Process(target=init_process, args=(rank, size, run_reduce))
        p.start()
        process.append(p)

    for p in process:
        p.join()


    