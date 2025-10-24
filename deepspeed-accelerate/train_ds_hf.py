from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)


from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch


def main():
    model_name = "distilbert-base-uncased"

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype = torch.float32,  # Use float32 to avoid mixed precision issues
        device_map = "auto",
        num_labels = 2,  # Binary classification
    )


    # Create synthetic data to avoid Hugging Face Hub issues
    from datasets import Dataset
    import random
    
    # Create synthetic text data
    texts = [
        "This is a great movie, I loved it!",
        "Terrible film, waste of time.",
        "Amazing story and acting.",
        "Boring and predictable plot.",
        "Excellent cinematography and direction.",
        "Poor script and bad acting.",
        "Wonderful experience, highly recommended.",
        "Disappointing and overrated.",
    ] * 25  # Repeat to get 200 samples
    
    labels = [1, 0, 1, 0, 1, 0, 1, 0] * 25  # Binary labels
    
    # Create dataset
    dataset = Dataset.from_dict({
        "text": texts,
        "label": labels
    })
    
    def tokenize_fn(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
    dataset = dataset.map(tokenize_fn, batched=True)
    

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_lin", "v_lin"],
        lora_dropout=0.1,
        bias = "none",
        task_type = "SEQ_CLS",
    )

    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir = "./outputs",
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        learning_rate = 2e-4,
        num_train_epochs = 1,
        logging_steps = 10,
        fp16 = False,  # Disable mixed precision to avoid gradient scaling issues
        # deepspeed = "ds_config_zero3.json",  # Commented out for Windows compatibility
        report_to = "none",
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = dataset,
        eval_dataset = dataset,
        tokenizer = tokenizer,
    )

    print("Training 1 epoch...")
    trainer.train()
    print("Training complete!")

if __name__ == "__main__":
    main()