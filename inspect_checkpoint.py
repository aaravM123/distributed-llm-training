import torch

checkpoint = torch.load("fsdp_checkpoint.pt", map_location="cpu")

print("\n Keys in checkpoint:")
for key in checkpoint.keys():
    for key in checkpoint.keys():
        print(f" - {key}")
print("\n Model State Dict Summary:")
for name, param in checkpoint["model"].items():
    print(f"{name:40s} | shape: {tuple(param.shape)}")
    break
