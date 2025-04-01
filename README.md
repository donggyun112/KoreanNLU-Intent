## Training

The repository includes comprehensive training scripts for:
1. Training from scratch
2. Fine-tuning existing models
3. Data augmentation
4. Handling class imbalance

### Training a New Model

```python
from intent_trainer import IntentClassificationSystem

# Initialize the system
system = IntentClassificationSystem()

# Load training data
train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = \
    system.load_training_data('example_training_data.csv')

# Train model
system.train(train_texts, train_labels, val_texts, val_labels, epochs=5, batch_size=16)

# Save model
system.save_model(version="my_new_model")
```

### Updating an Existing Model

```python
from intent_fine_tuner import AdditionalPatternTrainer

# Initialize the trainer
trainer = AdditionalPatternTrainer()

# Load pre-trained model
trainer.load_pretrained("intent_models/intent_model_best_model_epoch15_acc99.25.pt")

# Define new training data
train_texts = ["새로운 패턴 1", "새로운 패턴 2", ...]
train_labels = [1, 2, ...]  # Corresponding intent IDs

# Fine-tune model
best_epoch, best_accuracy = trainer.train_additional_pattern(
    train_texts,
    train_labels,
    epochs=5,
    batch_size=16,
    freeze_mode='partial'  # Options: 'partial', 'all', 'none'
)

# Save updated model
trainer.save_model(f"updated_model_epoch{best_epoch}_acc{best_accuracy:.2f}")
```

### Fine-tuning Options

The `intent_fine_tuner.py` module offers several fine-tuning strategies:

- **freeze_mode='partial'**: Freezes most BERT layers, only fine-tunes the last few layers
- **freeze_mode='all'**: Freezes all BERT layers, only trains the classifier head
- **freeze_mode='none'**: Fine-tunes the entire model

This allows for efficient adaptation to new patterns while preserving knowledge from the pre-trained model.# Korean Intent Classification System

This repository contains a Korean language intent classification system built with KoBERT. The system is designed to classify user utterances into five intent categories with a focus on media playback and search functionalities.

## Overview

The system uses a fine-tuned KoBERT model enhanced with dependency parsing features to accurately classify Korean language utterances into the following intents:

- `play.video`: Commands to play media content
- `search.video`: Requests to search for media content
- `resume.video`: Requests to continue previously watched content
- `set.channel.selected`: Commands to select or switch to a specific TV channel
- `undefined`: Utterances that don't match any of the defined intents

## Features

- High accuracy intent classification for Korean language
- Robust handling of various expression patterns
- Integration of syntactic information (dependency parsing) for improved accuracy
- Confidence threshold to identify uncertain classifications
- Comprehensive training pipeline with data augmentation capabilities

## Model Architecture

The model consists of:
- Pre-trained KoBERT base model (skt/kobert-base-v1)
- Dependency parsing features integration using Stanza
- Classification layer that outputs probabilities for each intent

The architecture leverages dependency parsing to improve classification by:
1. Parsing the input text using Stanza's Korean language pipeline
2. Extracting dependency relations (head-deprel pairs)
3. Combining original text with dependency information
4. Feeding the combined text through KoBERT
5. Classifying using the final layer probabilities

## Requirements

The project requires the following Python packages:

```
torch
transformers
kobert-tokenizer
stanza
pandas
numpy
scikit-learn
matplotlib
```

These dependencies are listed in the `requirements.txt` file and can be installed with:

```bash
pip install -r requirements.txt
```

Note: KoBERT tokenizer and Stanza may require additional resources for Korean language support.

## Installation

1. Clone this repository
2. Install required packages:
   ```
   pip install torch transformers kobert-tokenizer stanza pandas numpy scikit-learn matplotlib
   ```
3. Download the pre-trained model from Hugging Face:
   ```
   https://huggingface.co/dongkseo/Intent
   ```
4. Create a directory structure:
   ```
   mkdir -p intent_models
   ```
5. Place the downloaded model in the `intent_models` directory:
   ```
   mv intent_model_best_model_epoch15_acc99.25.pt intent_models/
   ```
   
## Usage

### Basic Usage

```python
from intent_inference import IntentClassificationSystem

# Initialize the system
system = IntentClassificationSystem()

# Load pre-trained model
system.load_model_file("intent_models/intent_model_best_model_epoch15_acc99.25.pt")

# Predict intent
text = "넷플릭스 영화 틀어줘"
intent = system.predict(text)
print(f"Intent: {intent}")

# Get prediction with confidence
logits, probs, intent = system.predict_with_probs(text, threshold=0.8)
print(f"Intent: {intent}, Confidence: {max(probs[0]):.4f}")
```

### Example Output

```
입력: '넷플릭스 영화 틀어줘'
예측 의도: play.video
로짓(Logits): [[10.2, -3.1, -2.8, -1.9, -8.3]]
확률(Probabilities): [[0.9982, 0.0002, 0.0004, 0.0011, 0.0001]]

입력: '유튜브 검색해줘'
예측 의도: search.video
로짓(Logits): [[-4.8, 11.9, -3.2, -2.1, -9.6]]
확률(Probabilities): [[0.0003, 0.9993, 0.0001, 0.0003, 0.0000]]

입력: '멈춘 곳부터 다시 보여줘'
예측 의도: resume.video
로짓(Logits): [[-3.7, -4.1, 10.8, -2.5, -7.4]]
확률(Probabilities): [[0.0004, 0.0002, 0.9987, 0.0006, 0.0001]]

입력: 'KBS 채널 틀어줘'
예측 의도: set.channel.selected
로짓(Logits): [[-2.9, -3.6, -4.3, 11.2, -8.7]]
확률(Probabilities): [[0.0006, 0.0003, 0.0001, 0.9989, 0.0001]]
```

## Batch Testing

The system supports batch prediction using an input file:

```python
from intent_inference import IntentClassificationSystem

# Initialize and load model
system = IntentClassificationSystem()
system.load_model_file("intent_models/intent_model_best_model_epoch15_acc99.25.pt")

# Load texts from file
texts = system.open_input_txt_file("input.txt")

# Predict intents
for text in texts:
    if not text:  # Skip empty lines
        continue
    logits, probs, intent = system.predict_with_probs(text, threshold=0.8)
    print(f"\n입력: '{text}'")
    print(f"예측 의도: {intent}")
    print(f"확률(Probabilities): {probs}")
```

## Data Augmentation Features

The model training pipeline in `model.py` includes sophisticated data augmentation techniques:

### Template-based Generation
- **Search Intent**: Generates variations using search keywords and patterns
  ```python
  # Example: "축구 경기 검색해봐", "최신 영화 찾아봐"
  search_keywords = ["축구 경기", "최신 영화", ...]
  search_patterns = ["{kw} 검색해봐", "{kw} 찾아봐", ...]
  ```

### Pattern Replacement
- **Play Intent**: Transforms expressions like "틀어줘" → "보여줘", "재생해줘", etc.
  ```python
  # "넷플릭스 영화 틀어줘" → "넷플릭스 영화 보여줘"
  ```

### Class Balancing
- **Undersampling**: Reduces majority classes (undefined, play.video)
- **Oversampling**: Increases minority classes with targeted counts
- **Smart Deduplication**: Removes duplicates from majority classes while preserving variety

### Example Pipeline
1. Load original data from CSV
2. Apply text augmentation via pattern replacement
3. Generate additional examples using templates
4. Balance classes (target ratios: undefined 20%, play.video 30%)
5. Perform targeted oversampling for minority classes
6. Remove duplicates while preserving class distribution
7. Split into train/val/test sets

## Data Format

Training data should be in CSV format with at least the following columns:
- `text`: The utterance text
- `intent`: The intent label (one of `play.video`, `search.video`, `resume.video`, `set.channel.selected`, or `undefined`)

## Model Performance

The pre-trained model included in this repository achieves:
- Accuracy: ~99.25% on test set
- Robust performance across all intent categories

## File Structure

The repository consists of the following files:

```
.
├── intent_inference.py        # Main inference system implementation
├── intent_trainer.py          # Complete model training pipeline with data augmentation
├── intent_fine_tuner.py       # Specialized trainer for fine-tuning models
├── example_training_data.csv  # Sample training data
├── example_training_data2.csv # Additional training dataset
├── input.txt                  # Example inputs for testing
├── intent_models/             # Directory for saved model files
│   └── intent_model_best_model_epoch15_acc99.25.pt  # Pre-trained model
└── requirements.txt           # Required packages
```

### Key Files

#### `intent_inference.py` (formerly `IntentClassifier.py`)
- Contains the main inference system implementation
- Defines the model architecture and prediction logic
- Handles model loading and provides APIs for intent classification

#### `intent_trainer.py` (formerly `model.py`)
- Implements a comprehensive training pipeline
- Defines `IntentClassifier` class (neural network architecture)
- Includes `IntentDataset` class for data handling
- Contains data augmentation functions:
  - Template-based generation for each intent type
  - Text augmentation with pattern replacements
  - Class balancing with undersampling/oversampling
- Implements the complete `IntentClassificationSystem` class with:
  - Training and evaluation functions
  - Model saving and loading
  - Confusion matrix visualization
  - Data preprocessing pipeline

#### `intent_fine_tuner.py` (formerly `model_update.py`)
- Contains the `AdditionalPatternTrainer` class for fine-tuning existing models
- Implements strategies for updating models with new patterns
- Provides partial model freezing capabilities to preserve knowledge
- Includes threshold-based prediction for handling uncertain cases
- Offers test functions for model evaluation

#### `example_training_data.csv` and `example_training_data2.csv`
- Sample datasets for training and testing the model
- Include text utterances and their corresponding intent labels

#### `input.txt`
- Contains sample inputs for batch testing the model

#### `requirements.txt`
- Lists all required Python packages and dependencies

## Additional Notes

- The system handles various Korean language patterns and expressions
- Intent prediction includes confidence scores to filter uncertain classifications
- The model has been optimized for media-related commands and queries


## Acknowledgments

- This model uses the KoBERT pre-trained model by SKT
- Dependency parsing is performed using the Stanza library
