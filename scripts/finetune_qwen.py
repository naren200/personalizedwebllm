#!/usr/bin/env python3
"""
Fine-tune Qwen 0.6B model on custom elevator pitch dataset using LoRA.
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import yaml
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_name: str = field(
        default="Qwen/Qwen-0.5B-Chat",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    trust_remote_code: bool = field(default=True, metadata={"help": "Trust remote code"})
    use_fast_tokenizer: bool = field(default=True, metadata={"help": "Use fast tokenizer"})


@dataclass
class DataArguments:
    """Arguments for data configuration."""
    dataset_path: str = field(metadata={"help": "Path to the training dataset"})
    max_seq_length: int = field(default=2048, metadata={"help": "Maximum sequence length"})
    validation_split: float = field(default=0.1, metadata={"help": "Validation split ratio"})


@dataclass
class LoRAArguments:
    """Arguments for LoRA configuration."""
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout"})
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
        metadata={"help": "LoRA target modules"}
    )


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def format_conversation(example: Dict) -> Dict:
    """Format conversation for Qwen chat template."""
    messages = [
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]}
    ]
    return {"messages": messages}


def prepare_dataset(data_args: DataArguments, tokenizer) -> Dataset:
    """Load and prepare the dataset."""
    logger.info(f"Loading dataset from {data_args.dataset_path}")
    
    # Load dataset
    if data_args.dataset_path.endswith('.json'):
        with open(data_args.dataset_path, 'r') as f:
            data = json.load(f)
        dataset = Dataset.from_list(data)
    else:
        dataset = load_dataset(data_args.dataset_path, split='train')
    
    # Format conversations
    dataset = dataset.map(format_conversation)
    
    # Split into train/validation
    if data_args.validation_split > 0:
        dataset = dataset.train_test_split(test_size=data_args.validation_split)
        train_dataset = dataset['train']
        eval_dataset = dataset['test']
    else:
        train_dataset = dataset
        eval_dataset = None
    
    logger.info(f"Training samples: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Validation samples: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


def tokenize_function(examples, tokenizer, max_seq_length):
    """Tokenize the examples."""
    texts = []
    for messages in examples["messages"]:
        # Apply Qwen chat template
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        texts.append(text)
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=max_seq_length,
        return_tensors=None,
    )
    
    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized


def main():
    """Main training function."""
    parser = HfArgumentParser((ModelArguments, DataArguments, LoRAArguments, TrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Load from JSON config file
        model_args, data_args, lora_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()
    
    # Load configuration from YAML if exists
    config = load_config("config/training_config.yaml")
    
    # Override with config values if available
    if config:
        for key, value in config.get("model", {}).items():
            if hasattr(model_args, key):
                setattr(model_args, key, value)
        
        for key, value in config.get("training", {}).items():
            if hasattr(training_args, key):
                setattr(training_args, key, value)
        
        for key, value in config.get("lora", {}).items():
            if hasattr(lora_args, key):
                setattr(lora_args, key, value)
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    
    logger.info(f"Training/evaluation parameters {training_args}")
    
    # Set seed
    torch.manual_seed(training_args.seed)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {model_args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=model_args.use_fast_tokenizer,
    )
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    logger.info(f"Loading model from {model_args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name,
        quantization_config=bnb_config,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Setup LoRA
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare dataset
    train_dataset, eval_dataset = prepare_dataset(data_args, tokenizer)
    
    # Tokenize datasets
    logger.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, data_args.max_seq_length),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    
    if eval_dataset:
        eval_dataset = eval_dataset.map(
            lambda x: tokenize_function(x, tokenizer, data_args.max_seq_length),
            batched=True,
            remove_columns=eval_dataset.column_names,
        )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Check for checkpoint
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif os.path.isdir(training_args.output_dir):
        checkpoint = get_last_checkpoint(training_args.output_dir)
    
    # Train
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=checkpoint)
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()
    trainer.save_state()
    
    # Save tokenizer
    tokenizer.save_pretrained(training_args.output_dir)
    
    logger.info(f"Training completed! Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()