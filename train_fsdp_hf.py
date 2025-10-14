import torch, torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    model_name = "meta-llama/Llama-2-7b-hf"

    print(f"[Rank {rank}] Initializing model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype= torch.bfloat16,
        device_map=None
    )

    auto_wrap = transformer_auto_wrap_policy
    model = FSDP(model, auto_wrap_policy=auto_wrap)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"[Rank {rank}] Model ready.")

if __name__ == "__main__":
    main()