import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from torch.cuda.amp import GradScaler, autocast

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--backend", type=str, default = "nccl")
args = parser.parse_args()

dist.init_process_group(backend = args.backend)
local_rank = int(os.environ.get("LOCAL_RANK",0))
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10,64),
            nn.ReLU(),
            nn.Linear(64,2)
        )
    def forward(self,x):
        return self.net(x)

model = TestModel().to(device)
model = FSDP(model)

x = torch.randn(64,10).to(device)
y = torch.randint(0,2,(64,)).to(device)
dataset = TensorDataset(x,y)
loader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True)

optimizer = optim.Adam(model.parameters(), lr = args.lr)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()

checkpoint_path = "fsdp_checkpoint.pt"
start_epoch = 0

if os.path.exists(checkpoint_path):
    map_location = {"cuda:%d" % 0: "cuda:%d" % (local_rank)}
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resumed training from epoch {start_epoch}")

import time
start_time = time.time()
torch.cuda.reset_peak_memory_stats()

for epoch in range(start_epoch, args.epochs):
    model.train()
    total_loss = 0.0

    # Reset timing and memory tracking for each epoch
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    for step, (inputs, labels) in enumerate(loader):
        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if step % 5 == 0 and dist.get_rank() == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

    # Compute epoch metrics
    avg_loss = total_loss / len(loader)
    elapsed = time.time() - start_time
    throughput = len(loader.dataset) / elapsed
    peak_mem = torch.cuda.max_memory_allocated() / 1e6

    if dist.get_rank() == 0:
        print(
            f"Epoch {epoch} finished. "
            f"Avg Loss: {avg_loss:.4f}, "
            f"Time: {elapsed:.2f}s, "
            f"Throughput: {throughput:.2f} samples/sec, "
            f"Peak Mem: {peak_mem:.2f} MB"
        )
        
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
        }, "fsdp_checkpoint.pt")
        print(f"[Rank 0] Check point saved for epoch {epoch}")

if dist.is_initialized():
    dist.destroy_process_group()
