import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import wandb
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3,32,3,padding=1)
        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*8*8,128)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,64*8*8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    wandb.init(project="ddp-toy")


    transform = transforms.Compose([transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])])

    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    if args.mode == "ddp":
        dist.init_process_group(backend = args.backend)

    if args.mode == "ddp":
        train_sampler = DistributedSampler(trainset)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, sampler=train_sampler, num_workers=2
        )
    else:
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=2
        )


    model = SimpleCNN().to(device)

    if args.mode == "dp":
        print("Running in DataParallel mode...")
        model = nn.DataParallel(model)
    elif args.mode == "ddp":
        print("Running in Distributed DataParallel mode...")
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank]
        )


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss = loss / args.grad_accum_steps
            loss.backward()

            if (i+1) % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            wandb.log({"loss": loss.item()})

            running_loss += loss.item()
            if i % 100 == 99:
                msg = f"[Epoch {epoch+1}, Step {i+1}] loss: {running_loss/100:.3f}"
                print(msg)
                with open("train.log", "a") as f:
                    f.write(msg + "\n")
                running_loss = 0.0

    print("Finished Training")
    
    if args.mode == "ddp":
        dist.destroy_process_group()

def main():
    import torch.distributed as dist

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"Hello from rank {rank} out of {world_size}")
        dist.barrier()
        print(f"Rank {rank} passed barrier")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--mode", choices=["single", "dp", "ddp"], default="single")
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=0)  
    parser.add_argument("--grad_accum_steps", type = int, default=1, help="Number of steps to accumulate gradients before updating.")

    args = parser.parse_args()
    main()
    train(args)
