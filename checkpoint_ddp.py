import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
import argparse
import os
import torch.distributed as dist

def collate_fn(batch):
    """Custom collate function to handle variable-length sequences"""
    # Get the maximum length in the batch
    max_len = max(len(item['input_ids']) for item in batch)
    
    # Pad sequences to the same length
    input_ids = []
    attention_masks = []
    labels = []
    
    for item in batch:
        # Convert to tensor if not already
        input_ids_tensor = torch.tensor(item['input_ids']) if not isinstance(item['input_ids'], torch.Tensor) else item['input_ids']
        attention_mask_tensor = torch.tensor(item['attention_mask']) if not isinstance(item['attention_mask'], torch.Tensor) else item['attention_mask']
        
        # Pad input_ids
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
        
        # Labels don't need padding
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
    model.to(device)

    if args.mode == "ddp":
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [device.index])

    elif args.mode == "dp" and torch.cuda.device_count()>1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)    
    

    optimizer = AdamW(model.parameters(), lr=args.lr)

    # --- Resume logic ---
    start_epoch = 0
    checkpoint_path = "checkpoint.pt"

    if os.path.exists(checkpoint_path):
        # map_location automatically matches current device (GPU or CPU)
        map_location = device if device.type == "cuda" else "cpu"
        ckpt = torch.load(checkpoint_path, map_location=map_location)

        model_to_load = model.module if hasattr(model, "module") else model
        model_to_load.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])

        # Move optimizer tensors to correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        start_epoch = ckpt["epoch"] + 1
        if args.mode != "ddp" or dist.get_rank() == 0:
            print(f"Resumed training from epoch {start_epoch}")
    else:
        if args.mode != "ddp" or dist.get_rank() == 0:
            print("Starting fresh training (no checkpoint found).")

    import time
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        model.train()
        total_loss = 0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 100 == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        elapsed = time.time() - start_time
        throughput = len(train_loader.dataset) / elapsed
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}, Time: {elapsed:.2f}s, Throughput: {throughput:.2f} samples/sec")

        if args.mode == "ddp" and dist.get_rank() == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            torch.save(checkpoint, "checkpoint.pt")
            print(f"[Rank 0] checkpoint saved for epoch {epoch+1}")

        # All ranks wait here (safe even if single GPU)
        if args.mode == "ddp":
            dist.barrier()


    if args.mode == "ddp":
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--mode", type = str, default = "single", choices = ["single", "dp", "ddp"], help = "Training mode: single GPU,DataParallel (dp), or DDP (ddp)")
    args = parser.parse_args()

    main(args) 
