#!/usr/bin/env python3
"""
Example: How to load HHEM model from local path

This example shows different ways to load the HHEM-2.1-Open model locally.
"""

import os
from dingo.model.rule.rule_hallucination_hhem import RuleHallucinationHHEM
from dingo.io import Data

def example_local_loading():
    """Example of loading HHEM model from local path"""
    
    # Method 1: Download model first, then use local path
    print("=== Method 1: Download model first ===")
    
    # First, download the model to a local directory
    # You can do this by running the model once with the default settings
    # or manually downloading it using transformers
    
    # Example local path (adjust to your actual path)
    local_model_path = "/path/to/your/local/hhem_model"
    
    # Check if local model exists
    if os.path.exists(local_model_path):
        print(f"✅ Local model found at: {local_model_path}")
        
        # Create test data
        test_data = Data(
            content="The sky is green and the grass is blue.",
            context="The sky is blue and the grass is green."
        )
        
        # Use local model
        result = RuleHallucinationHHEM.eval(test_data, model_path=local_model_path)
        print(f"Result: {result.score}")
        print(f"Error status: {result.error_status}")
    else:
        print(f"❌ Local model not found at: {local_model_path}")
        print("Please download the model first or adjust the path")

def download_model_locally():
    """Helper function to download model to local directory"""
    from transformers import AutoModelForSequenceClassification
    
    # Specify where you want to save the model
    local_path = "./hhem_model"
    
    print(f"Downloading HHEM model to: {local_path}")
    
    # Download and save model
    model = AutoModelForSequenceClassification.from_pretrained(
        'vectara/hallucination_evaluation_model',
        trust_remote_code=True
    )
    
    # Save to local directory
    model.save_pretrained(local_path)
    print(f"✅ Model saved to: {local_path}")
    
    return local_path

def example_with_config():
    """Example using configuration to set model path"""
    
    # You could also modify the class to use a configurable path
    # This would require modifying the dynamic_config or adding a class variable
    
    print("=== Method 2: Using configuration ===")
    
    # Example of how you might set a default local path
    # (This would require additional modifications to the class)
    
    # For now, you can set it manually in each call
    local_path = "./models/hhem_model"  # Adjust to your path
    
    if os.path.exists(local_path):
        test_data = Data(
            content="Paris is the capital of France.",
            context="Paris is the capital city of France."
        )
        
        result = RuleHallucinationHHEM.eval(test_data, model_path=local_path)
        print(f"Local model result: {result.score}")
    else:
        print(f"Model not found at {local_path}")

if __name__ == "__main__":
    print("HHEM Local Loading Examples")
    print("=" * 50)
    
    # Uncomment the method you want to try:
    
    # Method 1: Use existing local model
    example_local_loading()
    
    # Method 2: Download model first (uncomment to use)
    # local_path = download_model_locally()
    # example_local_loading()
    
    # Method 3: Using configuration
    # example_with_config()
