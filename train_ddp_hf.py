import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
import argparse

def collate_fn(batch):
    """Custom collate function to handle variable-length sequences"""
    # Get the maximum length in the batch
    max_len = max(len(item['input_ids']) for item in batch)
    
    # Pad sequences to the same length
    input_ids = []
    attention_masks = []
    labels = []
    
    for item in batch:
        # Pad input_ids
        padded_input_ids = item['input_ids'] + [0] * (max_len - len(item['input_ids']))
        input_ids.append(torch.tensor(padded_input_ids))
        
        # Pad attention_mask
        padded_attention_mask = item['attention_mask'] + [0] * (max_len - len(item['attention_mask']))
        attention_masks.append(torch.tensor(padded_attention_mask))
        
        # Labels don't need padding
        labels.append(item['label'])
    
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
        'label': torch.tensor(labels)
    }

def main(args):  
    dataset = load_dataset("ag_news")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True, max_length=128)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_loader = DataLoader(dataset["train"], batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
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
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    args = parser.parse_args()

    main(args) 
