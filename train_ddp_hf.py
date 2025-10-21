import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import argparse
import torch.distributed as dist
import csv, torch, os, time


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences"""
    # Get the maximum length in the batch
    max_len = max(len(item['input_ids']) for item in batch)
    
    # Pad sequences to the same length
    input_ids = []
    attention_masks = []
    labels = []
    
    for item in batch:
        input_ids_tensor = torch.tensor(item['input_ids']) if not isinstance(item['input_ids'], torch.Tensor) else item['input_ids']
        attention_mask_tensor = torch.tensor(item['attention_mask']) if not isinstance(item['attention_mask'], torch.Tensor) else item['attention_mask']
        
        if len(input_ids_tensor) < max_len:
            padding = torch.zeros(max_len - len(input_ids_tensor), dtype=input_ids_tensor.dtype)
            padded_input_ids = torch.cat([input_ids_tensor, padding])
        else:
            padded_input_ids = input_ids_tensor
        input_ids.append(padded_input_ids)
        
        # Pad attention_mask
        if len(attention_mask_tensor) < max_len:
            padding = torch.zeros(max_len - len(attention_mask_tensor), dtype=attention_mask_tensor.dtype)
            padded_attention_mask = torch.cat([attention_mask_tensor, padding])
        else:
            padded_attention_mask = attention_mask_tensor
        attention_masks.append(padded_attention_mask)
        
        labels.append(item['label'])
    
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
        'label': torch.tensor(labels)
    }

def main(args):  
    dataset = load_dataset("ag_news")

    dataset["train"] = dataset["train"].select(range(1000))
    dataset["test"] = dataset["test"].select(range(200))

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True, max_length=128)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    if args.mode == "ddp":
        dist.init_process_group(backend = "nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "ddp":
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset["train"])
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        dataset["train"], batch_size = args.batch_size, shuffle = shuffle, sampler = train_sampler, collate_fn = collate_fn
    )

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.auto_wrap:
        print("Using transformer _auto_wrap_policy to do FSDP wrapping.")
        auto_wrap_policy = transformer_auto_wrap_policy
        model = FSDP(model, auto_wrap_policy = auto_wrap_policy)

    if args.mode == "ddp" and not args.auto_wrap:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [device.index])

    elif args.mode == "dp" and torch.cuda.device_count()>1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)    
    

    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    start_step = 0
    if hasattr(args, "resume_step") and args.resume_step is not None:
        print(f"Resuming from step {args.resume_step}")
        start_step = args.resume_step

    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0

        grad_accumulation_steps = getattr(args, "grad_accumulation_steps", 1)
        optimizer.zero_grad()
        torch.cuda.reset_peak_memory_stats()

        for step, batch in enumerate(train_loader):
            if step < start_step:
                continue

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accumulation_steps
            loss.backward()
            total_loss += loss.item() * grad_accumulation_steps

            if (step+1) % grad_accumulation_steps == 0 or (step+1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            if (step+1) % 100 == 0:
                print(f"Epoch {epoch+1}, Step {step+1}, Loss: {loss.item()*grad_accumulation_steps:.4f}")

        avg_loss = total_loss / len(train_loader)
        epoch_elapsed = time.time() - epoch_start
        epoch_throughput = len(train_loader.dataset) / epoch_elapsed
        peak_mem = torch.cuda.max_memory_allocated() / 1e6
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}, Time: {epoch_elapsed:.2f}s, Throughput: {epoch_throughput:.2f} samples/sec, Peak Mem: {peak_mem:.0f} MB")

        if not dist.is_initialized() or dist.get_rank() == 0:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = f"checkpoints/epoch_{epoch+1}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, ckpt_path)
            print("Saved checkpoint to", {ckpt_path})

    mode = args.mode
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    
    os.makedirs("results", exist_ok=True)
    with open("results/benchmark_results.csv", "a", newline="") as f:
        writer = csv.writer(f)
        final_time = epoch_elapsed
        final_throughput = epoch_throughput

        os.makedirs("results", exist_ok=True)
        with open("results/benchmark_results.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([mode, n_gpus, batch_size, lr, epochs, final_time, final_throughput, peak_mem])


    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Time: {epoch_elapsed:.2f}s, Throughput: {epoch_throughput:.2f} samples/sec, Peak Mem: {peak_mem:.2f} MB")

    if args.mode == "ddp":
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--mode", type = str, default = "single", choices = ["single", "dp", "ddp"], help = "Training mode: single GPU,DataParallel (dp), or DDP (ddp)")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1)
    parser.add_argument("--resume_step", type=int, default=None, help="Resume training from a specific step")
    parser.add_argument("--auto_wrap", type=str, default=None, help="Enable transformer_auto_wrap_policy for FSDP testing")
    args = parser.parse_args()

    main(args) 
