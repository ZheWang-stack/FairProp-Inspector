import argparse
import json
import os
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import torch
from datasets import Dataset # type: ignore
from transformers import ( # type: ignore
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
)
from rich.console import Console
from rich.logging import RichHandler
import logging

# --- Configuration & Setup ---
console = Console()

def setup_logging():
    """Configures professional-grade logging with Rich."""
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )
    return logging.getLogger("fairprop")

logger = setup_logging()

@dataclass
class ModelConfig:
    """Configuration for the model architecture."""
    # modernbert-base is SOTA for efficient classification as of late 2024.
    # It supports 8192 context length and Flash Attention natively.
    feature_checkpoint: str = "answerdotai/ModernBERT-base" 
    max_length: int = 512  # Sufficient for property descriptions
    num_labels: int = 2

def load_data(file_path: str) -> Dataset:
    """
    Load JSON data and convert to HuggingFace Dataset.
    
    Args:
        file_path: Path to the JSON dataset file.
        
    Returns:
        A HuggingFace Dataset object ready for processing.
    """
    if not os.path.exists(file_path):
        logger.critical(f"Data file not found: {file_path}")
        sys.exit(1)
        
    logger.info(f"Loading data from [bold cyan]{file_path}[/bold cyan]...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data: List[Dict[str, Any]] = json.load(f)
    except json.JSONDecodeError:
        logger.critical("Failed to parse JSON data.")
        sys.exit(1)
    
    # Format for HF Dataset: list of dicts -> dict of lists
    hf_data = {
        "text": [item["text"] for item in data],
        "label": [item["label"] for item in data]
    }
    logger.info(f"Successfully loaded [bold green]{len(data)}[/bold green] examples.")
    return Dataset.from_dict(hf_data)

def compute_metrics(eval_pred: Any) -> Dict[str, float]:
    """Compute accuracy metrics for the trainer."""
    predictions, labels = eval_pred
    preds = predictions.argmax(-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}

def main():
    parser = argparse.ArgumentParser(description="Fine-tune ModernBERT for FHA Compliance.")
    parser.add_argument("--data", type=str, default="data/processed/seed_data.json", help="Path to training data")
    parser.add_argument("--output_dir", type=str, default="artifacts/model", help="Where to save the model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    args = parser.parse_args()

    console.rule("[bold purple]FairProp Inspector Training[/bold purple]")
    logger.info(f"Model Architecture: [bold magenta]{ModelConfig.feature_checkpoint}[/bold magenta]")
    
    # 1. Prepare Data
    dataset = load_data(args.data)
    # Split: 80% Train, 20% validation
    dataset = dataset.train_test_split(test_size=0.2, seed=42) # type: ignore

    # 2. Tokenization
    # ModernBERT requires a specific tokenizer. 
    # We use fast tokenizers for performance improvement in data preprocessing.
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(ModelConfig.feature_checkpoint)

    def preprocess_function(examples: Dict[str, List[str]]) -> Any:
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=ModelConfig.max_length,
            padding=False # Dynamic padding is handled by collator
        )

    logger.info("Tokenizing dataset...")
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    
    # Data Collator handles dynamic padding (pad to longest in batch, not max_length)
    # This acts as a significant speedup for variable length text.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 3. Model Initialization
    id2label = {0: "COMPLIANT", 1: "NON_COMPLIANT"}
    label2id = {"COMPLIANT": 0, "NON_COMPLIANT": 1}
    
    model = AutoModelForSequenceClassification.from_pretrained(
        ModelConfig.feature_checkpoint, 
        num_labels=ModelConfig.num_labels, 
        id2label=id2label, 
        label2id=label2id
    )

    # 4. Trainer Configuration
    # We advocate for bf16 (Bfloat16) if available for better stability than fp16.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        report_to="none", # We focus on local logs, enable 'wandb' for production runs
        logging_steps=10,
        bf16=use_bf16, # Hardware aware precision
        fp16=False if use_bf16 else torch.cuda.is_available(),
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch", # Fused optimizer for speed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"], # type: ignore
        eval_dataset=tokenized_datasets["test"], # type: ignore
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 5. Execution
    logger.info("🏋️‍♂️  Starting Training Loop...")
    trainer.train()
    
    # 6. Artifact Serialization
    logger.info(f"💾  Saving best model checkpoint to [bold yellow]{args.output_dir}[/bold yellow]...")
    trainer.save_model(args.output_dir)
    console.rule("[bold green]Training Complete[/bold green]")

if __name__ == "__main__":
    main()
