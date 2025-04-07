# Transformer for Question Answering (SQuAD)

This project implements a Transformer-based model for extractive question answering, trained on a subset of the Stanford Question Answering Dataset (SQuAD).

## Project Goal

The goal is to build and understand a Transformer model capable of finding the answer span within a given context paragraph based on a provided question. This involves:

1.  Loading and preprocessing the SQuAD dataset.
2.  Building a Transformer architecture (Encoder-style) from scratch using TensorFlow/Keras.
3.  Implementing custom components like Positional Encoding and Multi-Head Self-Attention.
4.  Training the model to predict the start and end token indices of the answer span.
5.  Evaluating the model using standard QA metrics like Exact Match (EM) and F1-score.

## Core Components

*   **`train_squad.py`**: The main script to load data, build the model, train it, evaluate (basic loss/accuracy), and save it.
*   **`utils/squad_preprocessing.py`**: Handles loading the `squad.json` file, training and utilizing the `SubwordTokenizer`, formatting inputs (`[CLS] question [SEP] context [SEP]`), and finding answer token spans.
*   **`utils/subword_tokenizer.py`**: Implements a WordPiece subword tokenizer using the `tokenizers` library, trained from scratch on the SQuAD data. Includes methods for encoding, decoding, and managing special tokens.
*   **`models/qa_transformer.py`**: Defines the overall Transformer architecture using custom components, outputting start and end logits for each token.
*   **`components/`**: Contains the building blocks:
    *   `attention.py`: `MultiHeadSelfAttention` layer.
    *   `positional_encoding.py`: `PositionalEncoding` layer.
    *   `transformer_block.py`: Combines attention and feed-forward layers into a standard `TransformerBlock`.
*   **`utils/evaluation.py`**: Contains helper functions for timing training and calculating SQuAD evaluation metrics (EM, F1 - requires integration into the training script for full use).
*   **`squad.json`**: The dataset file (not included in the repo, should be downloaded separately).

## Setup

1.  **Clone the repository.**
2.  **Create a Python environment** (e.g., using Conda or venv).
3.  **Install dependencies:**
    ```bash
    pip install tensorflow numpy tokenizers
    ```
    *(Ensure you have a version of TensorFlow compatible with your GPU and CUDA/cuDNN if using GPU acceleration).* 
4.  **Download the SQuAD dataset:** Obtain the `squad.json` file (e.g., SQuAD v2.0 `train-v2.0.json`) and place it in the project's root directory.

## Running the Training

Execute the main training script:

```bash
python train_squad.py
```

This will:
*   Load and preprocess the data.
*   Train the subword tokenizer on the text corpus.
*   Build the QA Transformer model with specified hyperparameters (e.g., `D_MODEL`, `NUM_HEADS`, `NUM_LAYERS`).
*   Train the model for a set number of `EPOCHS`.
*   Print basic validation loss and accuracy.
*   Save the trained model to the `squad_transformer_model` directory.

**Note:** Training was developed and tested using a single NVIDIA GeForce RTX 3080/4070 GPU.

## Hyperparameters

Key hyperparameters can be adjusted directly in `train_squad.py`:

*   `VOCAB_SIZE`: Maximum vocabulary size for training the subword tokenizer.
*   `SEQUENCE_LEN`: Fixed length for input sequences (padding/truncation).
*   `D_MODEL`: Embedding dimension / Transformer hidden size.
*   `NUM_HEADS`: Number of attention heads.
*   `DFF`: Dimension of the feed-forward network inner layer.
*   `NUM_LAYERS`: Number of Transformer blocks.
*   `DROPOUT_RATE`: Dropout rate used in the Transformer blocks.
*   `BATCH_SIZE`: Number of examples per training step.
*   `EPOCHS`: Number of passes through the training data.
*   `MAX_SAMPLES`: Maximum number of SQuAD examples to load (for faster experimentation).

## Further Development / Evaluation

*   Implement the full EM/F1 evaluation loop in `train_squad.py` using the functions provided in `utils/evaluation.py`.
*   Experiment with different pre-trained tokenizers (e.g., from Hugging Face Hub) or other subword algorithms (like BPE or SentencePiece).
*   Tune hyperparameters and use the full dataset (`MAX_SAMPLES = None`) for better performance.
*   Implement learning rate schedules.
*   Add detailed error analysis of model predictions.

## Experiment Tracking with Weights & Biases

This project uses [Weights & Biases (W&B)](https://wandb.ai/) for experiment tracking, visualization, and logging.

### Features Tracked:

*   **Hyperparameters:** All hyperparameters defined in `train_squad.py` are logged to W&B, allowing easy comparison between runs.
*   **Metrics:** Training and validation loss, accuracy, Exact Match (EM), and F1 score are tracked throughout training and logged at the end.
*   **Model Checkpoints:** (If configured) Model weights can be saved as W&B artifacts.
*   **System Metrics:** GPU/CPU utilization, memory usage, etc., are automatically tracked.
*   **Validation Examples:** A table comparing true answers and predicted answers for a subset of the validation set is logged.

### Setup:

1.  **Install W&B:**
    ```bash
    pip install wandb
    ```
2.  **Login:**
    ```bash
    wandb login
    ```
    You'll need a free W&B account. Follow the prompts to authenticate.
3.  **Run Training:**
    Simply run the training script as usual:
    ```bash
    python train_squad.py
    ```
    A new run will be automatically created in your W&B project (default is `squad-transformer-qa`, you can change this in `train_squad.py`). You'll see a link to the run page in your terminal output.

Visit the W&B dashboard linked in your terminal to monitor your training runs in real-time. 