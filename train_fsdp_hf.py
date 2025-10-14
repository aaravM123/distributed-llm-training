import torch, torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from functools import partial
import os

def main():
    # Use 'gloo' backend for Windows, 'nccl' for Linux with CUDA
    backend = "gloo" if os.name == "nt" else "nccl"
    dist.init_process_group(backend)
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Only set CUDA device if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    print(f"[Rank {rank}] Initializing model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=None
    )
    model.to(device)

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={MistralDecoderLayer},
    )
    model = FSDP(model, auto_wrap_policy=auto_wrap_policy)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"[Rank {rank}] Model ready.")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()