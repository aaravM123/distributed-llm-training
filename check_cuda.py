import torch
from transformers import AutoModelForCausalLM

print("CUDA / Torch Environment Check")
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("CUDA Version:", torch.version.cuda)
print("Torch Version:", torch.__version__)

print("\nLoading Small Model Test (OPT-350M)")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
print("Model loaded successfully!")
num_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Number of parameters: {num_params:.1f}M")