import re
import random
from typing import List, Dict
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

# Import augmentation functions from create_rlhf.py
from utils.create_rlhf import augment_negative

def print_example_pairs(df: pd.DataFrame, num_examples: int = 3):
    """Print example pairs in a nicely formatted way."""
    print("\n" + "="*80)
    print("EXAMPLE PAIRS FROM THE DATASET")
    print("="*80 + "\n")
    
    # Randomly select examples
    examples = df.sample(n=min(num_examples, len(df)))
    
    for idx, row in examples.iterrows():
        print(f"Example {idx + 1}:")
        print("-"*40)
        print("Instruction:")
        print(f"  {row['instruction']}")
        print("\nPositive Response:")
        print(f"  {row['positive']}")
        print("\nNegative Response:")
        print(f"  {row['negative']}")
        print("\n" + "="*80 + "\n")

def process_dataset():
    # Load the medical meadow alpaca dataset
    dataset = load_dataset("monology/medical_meadow_alpaca")
    
    # Initialize lists to store positive and negative examples
    instructions = []
    positive_outputs = []
    negative_outputs = []
    
    # Process each example
    for example in tqdm(dataset['train']):
        # Get the instruction, input and output
        instruction = example['instruction']
        input_text = example['input']
        positive_output = example['output']
        
        # Combine instruction and input with a clear separator
        combined_instruction = f"{instruction}\n\nContext:\n{input_text}" if input_text else instruction
        
        # Generate negative example using our augmentation pipeline
        negative_output = augment_negative(positive_output)
        
        # Append to lists
        instructions.append(combined_instruction)
        positive_outputs.append(positive_output)
        negative_outputs.append(negative_output)
    
    # Create DataFrame
    df = pd.DataFrame({
        'instruction': instructions,
        'positive': positive_outputs,
        'negative': negative_outputs
    })
    
    # Save to CSV
    output_file = 'medical_rlhf_pairs.csv'
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\nProcessed {len(instructions)} examples and saved to {output_file}")
    
    # Print example pairs
    print_example_pairs(df)
