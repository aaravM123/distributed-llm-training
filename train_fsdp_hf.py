import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import argeparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--backend", type=str, default = "nccl")
args = parser.parse_args()

dist.init_process_group(backend = args.backend)
local_rank = int(os.environ("LOCAL_RANK",0))
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

for epoch in range(args.epochs):
    for step, (inputs, labels) in enumerate(loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if step % 5 == 0 and dist.get_rank() == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

dist.destroy_process_group()

if dist.get_rank() == 0:
    print("Peak Memory Usage: ", torch.cuda.max_memory_allocated() / 1e6, "MB")