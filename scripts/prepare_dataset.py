#!/usr/bin/env python3
"""
Prepare and process dataset for Qwen fine-tuning.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_data(input_path: str) -> List[Dict]:
    """Load data from various formats."""
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    logger.info(f"Loading data from: {input_path}")
    
    if input_path.suffix.lower() == '.json':
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif input_path.suffix.lower() == '.jsonl':
        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    elif input_path.suffix.lower() in ['.csv', '.tsv']:
        separator = '\t' if input_path.suffix.lower() == '.tsv' else ','
        df = pd.read_csv(input_path, sep=separator)
        data = df.to_dict('records')
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    logger.info(f"Loaded {len(data)} records")
    return data


def standardize_format(data: List[Dict], input_col: str = None, output_col: str = None) -> List[Dict]:
    """Standardize data format to input/output pairs."""
    standardized = []
    
    # Auto-detect column names if not provided
    if data and isinstance(data[0], dict):
        columns = list(data[0].keys())
        
        if input_col is None:
            # Common input column names
            for col in ['input', 'question', 'prompt', 'user', 'instruction']:
                if col in columns:
                    input_col = col
                    break
        
        if output_col is None:
            # Common output column names
            for col in ['output', 'answer', 'response', 'assistant', 'completion']:
                if col in columns:
                    output_col = col
                    break
    
    if input_col is None or output_col is None:
        logger.warning(f"Could not auto-detect columns. Available: {columns}")
        logger.warning("Please specify --input_col and --output_col")
        return data
    
    logger.info(f"Using input column: '{input_col}', output column: '{output_col}'")
    
    for item in data:
        if input_col in item and output_col in item:
            standardized.append({
                'input': str(item[input_col]).strip(),
                'output': str(item[output_col]).strip()
            })
        else:
            logger.warning(f"Skipping item missing required columns: {item}")
    
    logger.info(f"Standardized {len(standardized)} records")
    return standardized


def filter_data(data: List[Dict], min_length: int = 10, max_length: int = 2048) -> List[Dict]:
    """Filter data based on length constraints."""
    filtered = []
    
    for item in data:
        input_len = len(item['input'])
        output_len = len(item['output'])
        
        if (min_length <= input_len <= max_length and 
            min_length <= output_len <= max_length):
            filtered.append(item)
        else:
            logger.debug(f"Filtered out item: input_len={input_len}, output_len={output_len}")
    
    logger.info(f"Filtered data: {len(filtered)}/{len(data)} records kept")
    return filtered


def augment_data(data: List[Dict], augment_prompts: List[str] = None) -> List[Dict]:
    """Augment data with additional prompt variations."""
    if augment_prompts is None:
        augment_prompts = [
            "Tell me about yourself.",
            "What's your elevator pitch?",
            "Describe your background.",
            "What do you do?",
            "Give me a brief introduction.",
            "What's your professional summary?",
            "How would you introduce yourself?",
            "What's your story?",
            "Tell me about your experience.",
            "What makes you unique?"
        ]
    
    augmented = []
    
    for item in data:
        # Keep original
        augmented.append(item)
        
        # Add variations for elevator pitch data
        if any(keyword in item['input'].lower() for keyword in ['elevator', 'pitch', 'introduction', 'about yourself']):
            for prompt in augment_prompts[:3]:  # Limit to avoid too much augmentation
                if prompt.lower() not in item['input'].lower():
                    augmented.append({
                        'input': prompt,
                        'output': item['output']
                    })
    
    logger.info(f"Augmented data: {len(data)} -> {len(augmented)} records")
    return augmented


def create_example_dataset(output_path: str, num_examples: int = 50) -> None:
    """Create an example elevator pitch dataset."""
    example_data = [
        {
            "input": "Tell me about yourself",
            "output": "I'm a software engineer with 5 years of experience in full-stack web development. I specialize in building scalable applications using modern frameworks like React and Node.js. I'm passionate about creating user-friendly solutions that solve real-world problems and have successfully led teams to deliver high-impact projects."
        },
        {
            "input": "What's your elevator pitch?",
            "output": "I'm a data scientist who transforms complex business challenges into actionable insights. With a background in machine learning and 3 years of experience at Fortune 500 companies, I've helped increase revenue by 25% through predictive analytics and automated decision-making systems."
        },
        {
            "input": "Describe your background",
            "output": "I'm a product manager with expertise in both B2B and B2C products. I've launched 10+ successful products that generated over $2M in revenue. My approach combines data-driven decision making with user-centered design to create products that customers love and businesses profit from."
        },
        {
            "input": "What do you do?",
            "output": "I'm a digital marketing specialist focused on growth hacking and conversion optimization. I help startups and established companies scale their online presence through strategic content marketing, SEO, and performance advertising. My campaigns have consistently delivered 3x ROI improvements."
        },
        {
            "input": "Give me a brief introduction",
            "output": "I'm a UX designer who bridges the gap between user needs and business goals. With 4 years of experience in design thinking and user research, I create intuitive interfaces that improve user satisfaction by 40% while driving business metrics. I believe great design is invisible but impactful."
        },
        {
            "input": "What's your professional summary?",
            "output": "I'm a cybersecurity expert specializing in threat detection and incident response. I've protected organizations from advanced persistent threats and reduced security incidents by 60% through proactive monitoring and team training. I'm passionate about making cybersecurity accessible and effective for all businesses."
        },
        {
            "input": "How would you introduce yourself?",
            "output": "I'm a sales professional who builds lasting relationships and drives revenue growth. In my 6 years of B2B sales experience, I've consistently exceeded quotas by 150% and generated $5M+ in pipeline. I believe in consultative selling and helping clients achieve their business objectives."
        },
        {
            "input": "What's your story?",
            "output": "I'm a project manager who thrives on turning chaos into order. I've successfully managed cross-functional teams of 20+ people and delivered complex projects on time and under budget. My secret is clear communication, stakeholder alignment, and a focus on outcomes rather than outputs."
        },
        {
            "input": "Tell me about your experience",
            "output": "I'm a financial analyst with expertise in investment research and portfolio management. I've analyzed $100M+ in assets and helped clients achieve 12% average returns through strategic investment decisions. I combine quantitative analysis with market intuition to identify undervalued opportunities."
        },
        {
            "input": "What makes you unique?",
            "output": "I'm a creative director who combines artistic vision with business strategy. I've led rebranding initiatives that increased brand recognition by 80% and customer engagement by 200%. My unique approach blends traditional design principles with cutting-edge technology to create memorable brand experiences."
        }
    ]
    
    # Extend with variations if needed
    while len(example_data) < num_examples:
        base_examples = example_data[:10]  # Use first 10 as base
        for example in base_examples:
            if len(example_data) >= num_examples:
                break
            # Create variation with different prompt
            prompts = ["Describe yourself", "What's your background?", "Tell me your story"]
            for prompt in prompts:
                if len(example_data) >= num_examples:
                    break
                example_data.append({
                    "input": prompt,
                    "output": example["output"]
                })
    
    # Save example dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(example_data[:num_examples], f, indent=2, ensure_ascii=False)
    
    logger.info(f"Created example dataset with {num_examples} examples: {output_path}")


def split_dataset(data: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1) -> tuple:
    """Split dataset into train/validation/test sets."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")
    
    # First split: train vs (val + test)
    train_data, temp_data = train_test_split(
        data, 
        test_size=(val_ratio + test_ratio), 
        random_state=42,
        shuffle=True
    )
    
    # Second split: val vs test
    if test_ratio > 0:
        val_size = val_ratio / (val_ratio + test_ratio)
        val_data, test_data = train_test_split(
            temp_data, 
            test_size=(1 - val_size), 
            random_state=42,
            shuffle=True
        )
    else:
        val_data = temp_data
        test_data = []
    
    logger.info(f"Dataset split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    return train_data, val_data, test_data


def save_dataset(data: List[Dict], output_path: str, format: str = 'json') -> None:
    """Save dataset in specified format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == 'json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    elif format.lower() == 'jsonl':
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    else:
        raise ValueError(f"Unsupported output format: {format}")
    
    logger.info(f"Saved {len(data)} records to: {output_path}")


def main():
    """Main data processing function."""
    parser = argparse.ArgumentParser(description="Prepare dataset for Qwen fine-tuning")
    parser.add_argument("--input", required=True, help="Input data file path")
    parser.add_argument("--output", required=True, help="Output directory path")
    parser.add_argument("--input_col", help="Input column name (auto-detected if not provided)")
    parser.add_argument("--output_col", help="Output column name (auto-detected if not provided)")
    parser.add_argument("--min_length", type=int, default=10, help="Minimum text length")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum text length")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test set ratio")
    parser.add_argument("--format", choices=['json', 'jsonl'], default='json', help="Output format")
    parser.add_argument("--augment", action="store_true", help="Augment data with prompt variations")
    parser.add_argument("--create_example", action="store_true", help="Create example dataset")
    parser.add_argument("--num_examples", type=int, default=50, help="Number of examples to create")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create example dataset if requested
    if args.create_example:
        example_path = output_dir / "example_dataset.json"
        create_example_dataset(str(example_path), args.num_examples)
        logger.info(f"Example dataset created at: {example_path}")
        logger.info("You can use this as a template for your own dataset.")
        return
    
    # Load and process data
    try:
        data = load_data(args.input)
        
        # Standardize format
        data = standardize_format(data, args.input_col, args.output_col)
        
        # Filter data
        data = filter_data(data, args.min_length, args.max_length)
        
        # Augment data if requested
        if args.augment:
            data = augment_data(data)
        
        # Split dataset
        train_data, val_data, test_data = split_dataset(
            data, args.train_ratio, args.val_ratio, args.test_ratio
        )
        
        # Save datasets
        save_dataset(train_data, output_dir / f"train.{args.format}", args.format)
        if val_data:
            save_dataset(val_data, output_dir / f"validation.{args.format}", args.format)
        if test_data:
            save_dataset(test_data, output_dir / f"test.{args.format}", args.format)
        
        # Save combined dataset
        save_dataset(data, output_dir / f"combined.{args.format}", args.format)
        
        # Create dataset info
        info = {
            "total_samples": len(data),
            "train_samples": len(train_data),
            "validation_samples": len(val_data),
            "test_samples": len(test_data),
            "format": args.format,
            "min_length": args.min_length,
            "max_length": args.max_length,
            "augmented": args.augment,
            "splits": {
                "train": args.train_ratio,
                "validation": args.val_ratio,
                "test": args.test_ratio
            }
        }
        
        with open(output_dir / "dataset_info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info("🎉 Dataset preparation completed successfully!")
        logger.info(f"Processed files saved to: {output_dir}")
        logger.info("Dataset is ready for fine-tuning!")
        
    except Exception as e:
        logger.error(f"❌ Dataset preparation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()