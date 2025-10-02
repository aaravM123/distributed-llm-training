import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

def main():
    dataset = load_dataset("ag_news")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True, max_length = 128)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_loader  = DataLoader(dataset["train"], batch_size=16, shuffle=True)

    for batch in train_loader:
        print(batch["input_ids"].shape, batch["attention_mask"].shape, batch["label"].shape)
        break

if __name__ == "__main__":
    main()