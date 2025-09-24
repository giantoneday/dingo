# Loading HHEM Model from Local Path

This guide explains how to load the HHEM-2.1-Open model from a local directory instead of downloading it from Hugging Face each time.

## Quick Start

### 1. Download the Model Locally

First, download the model to a local directory:

```python
from transformers import AutoModelForSequenceClassification

# Download and save the model
model = AutoModelForSequenceClassification.from_pretrained(
    'vectara/hallucination_evaluation_model',
    trust_remote_code=True
)

# Save to local directory
local_path = "./hhem_model"
model.save_pretrained(local_path)
print(f"Model saved to: {local_path}")
```

### 2. Use Local Model in Your Code

```python
from dingo.model.rule.rule_hallucination_hhem import RuleHallucinationHHEM
from dingo.io import Data

# Create test data
test_data = Data(
    content="The sky is green and the grass is blue.",
    context="The sky is blue and the grass is green."
)

# Use local model
local_model_path = "./hhem_model"
result = RuleHallucinationHHEM.eval(test_data, model_path=local_model_path)

print(f"Hallucination score: {result.score}")
print(f"Error status: {result.error_status}")
```

## Available Methods

The modified `RuleHallucinationHHEM` class now supports local loading in all its methods:

### 1. `eval()` - Single evaluation
```python
result = RuleHallucinationHHEM.eval(input_data, model_path="/path/to/local/model")
```

### 2. `evaluate_with_detailed_output()` - Detailed analysis
```python
detailed_result = RuleHallucinationHHEM.evaluate_with_detailed_output(
    input_data, 
    model_path="/path/to/local/model"
)
```

### 3. `batch_evaluate()` - Batch processing
```python
results = RuleHallucinationHHEM.batch_evaluate(
    data_list, 
    model_path="/path/to/local/model"
)
```

### 4. `load_model()` - Direct model loading
```python
RuleHallucinationHHEM.load_model("/path/to/local/model")
```

## Key Parameters

- **`model_path`**: Path to the local model directory
- **`local_files_only=True`**: Ensures the model is loaded only from local files (no internet download)
- **`trust_remote_code=True`**: Required for HHEM model (contains custom code)

## Benefits of Local Loading

1. **Faster startup**: No need to download model each time
2. **Offline usage**: Works without internet connection
3. **Version control**: Use specific model versions
4. **Custom models**: Load your own fine-tuned versions
5. **Reduced bandwidth**: No repeated downloads

## Directory Structure

After downloading, your local model directory should contain:

```
hhem_model/
├── config.json
├── pytorch_model.bin (or model.safetensors)
├── tokenizer.json
├── tokenizer_config.json
├── vocab.txt
└── [other model files]
```

## Error Handling

The code includes proper error handling for:

- Missing local model files
- Invalid model paths
- Corrupted model files
- Import errors (missing transformers library)

## Example: Complete Workflow

```python
import os
from dingo.model.rule.rule_hallucination_hhem import RuleHallucinationHHEM
from dingo.io import Data

def main():
    # Set your local model path
    local_model_path = "./models/hhem_model"
    
    # Check if model exists
    if not os.path.exists(local_model_path):
        print(f"Model not found at {local_model_path}")
        print("Please download the model first using the download script")
        return
    
    # Create test data
    test_data = Data(
        content="The Earth is flat and the moon is made of cheese.",
        context="The Earth is round and the moon is made of rock."
    )
    
    # Evaluate using local model
    result = RuleHallucinationHHEM.eval(test_data, model_path=local_model_path)
    
    # Print results
    print(f"Hallucination Score: {result.score:.3f}")
    print(f"Threshold: {RuleHallucinationHHEM.dynamic_config.threshold}")
    print(f"Hallucination Detected: {result.error_status}")
    
    if result.reason:
        print("\nDetailed Analysis:")
        print(result.reason[0])

if __name__ == "__main__":
    main()
```

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure the path is correct and contains all model files
2. **Permission errors**: Check file permissions on the model directory
3. **Memory issues**: The model requires ~600MB RAM
4. **Import errors**: Install transformers: `pip install transformers`

### Verification

To verify your local model works:

```python
# Test local model loading
try:
    RuleHallucinationHHEM.load_model("/path/to/your/model")
    print("✅ Local model loaded successfully")
except Exception as e:
    print(f"❌ Error loading local model: {e}")
```

## Performance Notes

- **First load**: May take 10-30 seconds depending on your system
- **Subsequent loads**: Much faster (model is cached in memory)
- **Memory usage**: ~600MB RAM when loaded
- **Inference speed**: ~1.5 seconds for 2k tokens on modern CPU
