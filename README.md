# Korean Intent Classification System

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
- Pre-trained KoBERT base model
- Dependency parsing features integration using Stanza
- Classification layer that outputs probabilities for each intent

## Requirements

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
   
## Usage

### Basic Usage

```python
from intent_classification import IntentClassificationSystem

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

### Batch Prediction

```python
sample_texts = [
    "넷플릭스 영화 틀어줘",
    "유튜브 검색해줘",
    "멈춘 곳부터 다시 보여줘",
    "KBS 채널 틀어줘"
]

for txt in sample_texts:
    logits, probs, intent = system.predict_with_probs(txt)
    print(f"Text: '{txt}', Intent: {intent}")
```

## Training

The repository includes comprehensive training scripts for:
1. Training from scratch
2. Fine-tuning existing models
3. Data augmentation
4. Handling class imbalance

To train a new model:

```python
from intent_training import IntentClassificationSystem

# Initialize the system
system = IntentClassificationSystem()

# Load training data
train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = \
    system.load_training_data('training_data.csv')

# Train model
system.train(train_texts, train_labels, val_texts, val_labels, epochs=5, batch_size=16)

# Save model
system.save_model(version="my_new_model")
```

## Data Format

Training data should be in CSV format with at least the following columns:
- `text`: The utterance text
- `intent`: The intent label (one of `play.video`, `search.video`, `resume.video`, `set.channel.selected`, or `undefined`)

## Model Performance

The pre-trained model included in this repository achieves:
- Accuracy: ~99.25% on test set
- Robust performance across all intent categories

## Additional Notes

- The system handles various Korean language patterns and expressions
- Intent prediction includes confidence scores to filter uncertain classifications
- The model has been optimized for media-related commands and queries

## Acknowledgments

- This model uses the KoBERT pre-trained model by SKT
- Dependency parsing is performed using the Stanza library