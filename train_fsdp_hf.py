import torch, torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from functools import partial
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
import os

def main():
    backend = "gloo" if os.name == "nt" else "nccl"
    dist.init_process_group(backend)
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Enable TF32 for better performance on Ampere GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    # NOTE: Increase --nproc_per_node to >1 to activate true sharding.
    # With 1 GPU, FSDP runs in NO_SHARD mode (no real memory savings).
    model_name = "meta-llama/Llama-2-7b-hf"
    # For small GPU testing, use: model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-token"

    print(f"[Rank {rank}] Initializing model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=None
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    print(f"[Rank {rank}] LoRA applied to model.")
    model.to(device)  # Move to GPU before FSDP
    
    # Ensure all parameters have the same dtype for FSDP
    for param in model.parameters():
        param.data = param.data.to(torch.bfloat16)

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )
    model = FSDP(model, auto_wrap_policy=auto_wrap_policy, device_id=local_rank, use_orig_params=True)
    print(f"[Rank {rank}] ✅ Model wrapped with FSDP and ready for training.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set padding token for Llama tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if dist.get_rank() == 0:
        print("[Data] Loading WikiText-2 small subset...")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    texts = dataset["train"]["text"][:200]

    enc = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)

    train_loader = DataLoader(
        list(zip(input_ids, attn_mask)),
        batch_size=1,
        shuffle=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr = 2e-5)

    model.train()
    for epoch in range(1):
        torch.cuda.reset_peak_memory_stats()
        for i, (input_ids, attn_mask) in enumerate(train_loader):
            input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    labels=input_ids
                )
                loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0 and dist.get_rank() == 0:
                print(f"Step {i}, Loss: {loss.item():.4f}")
            
            # Save checkpoint every 50 steps to avoid losing progress
            if i % 50 == 0 and i > 0 and dist.get_rank() == 0:
                try:
                    # Save only the LoRA parameters to avoid FSDP issues
                    lora_state_dict = {k: v for k, v in model.state_dict().items() if 'lora' in k}
                    torch.save(lora_state_dict, f"fsdp_lora_checkpoint_step_{i}.pt")
                    print(f"[Checkpoint] LoRA parameters saved at step {i}")
                except Exception as e:
                    print(f"[Warning] Failed to save checkpoint at step {i}: {e}")
        
        peak = torch.cuda.max_memory_allocated() / 1024**2
        if dist.get_rank() == 0:
            print(f"Epoch {epoch}, Peak Memory: {peak:.2f}MB")
    
    # Final save with better error handling
    if dist.get_rank() == 0:
        try:
            # Save only LoRA parameters to avoid FSDP state dict issues
            lora_state_dict = {k: v for k, v in model.state_dict().items() if 'lora' in k}
            torch.save(lora_state_dict, "fsdp_lora_checkpoint_final.pt")
            print("[Checkpoint] Final LoRA parameters saved as fsdp_lora_checkpoint_final.pt")
            
            # Also save a backup
            torch.save(lora_state_dict, "fsdp_lora_checkpoint_backup.pt")
            print("[Checkpoint] Backup saved as fsdp_lora_checkpoint_backup.pt")
        except Exception as e:
            print(f"[Error] Failed to save final checkpoint: {e}")
            # Try to save just the optimizer state as fallback
            try:
                torch.save(optimizer.state_dict(), "fsdp_optimizer_state.pt")
                print("[Checkpoint] Optimizer state saved as fallback")
            except Exception as e2:
                print(f"[Error] Failed to save optimizer state: {e2}")

    # Better cleanup with timeout handling
    try:
        # Give processes time to sync before cleanup
        dist.barrier(timeout=torch.distributed.constants.default_pg_timeout)
        dist.destroy_process_group()
        print("[Cleanup] Process group destroyed successfully")
    except Exception as e:
        print(f"[Warning] Process group cleanup failed: {e}")
        # Force cleanup
        try:
            dist.destroy_process_group()
        except:
            pass

if __name__ == "__main__":
    main()