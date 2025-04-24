# Transformer for Question Answering on SQuAD

This project aims to train a transformer model from scratch for extractive question answering using the SQuAD dataset. The goal is to understand the components of transformer models and compare a custom implementation against potentially fine-tuning pre-trained models later.


---


## Project Goal

The goal is to build and understand a Transformer model capable of finding the answer span within a given context paragraph based on a provided question. This involves:

1.  Loading and preprocessing the SQuAD dataset.
2.  Building a Transformer architecture (Encoder-style) from scratch using TensorFlow/Keras.
3.  Implementing custom components like Positional Encoding and Multi-Head Self-Attention.
4.  Training the model to predict the start and end token indices of the answer span.
5.  Evaluating the model using standard QA metrics like Exact Match (EM) and F1-score.

## Core Components

*   **`train_squad.py`**: The main script to load data, build the model, train it, evaluate (basic loss/accuracy), and save it.
*   **`utils/hf_squad_preprocessing.py`**: Handles loading the SQuAD dataset from Hugging Face, utilizing the Hugging Face tokenizer, formatting inputs (`[CLS] question [SEP] context [SEP]`), and finding answer token spans.
*   **`models/qa_transformer.py`**: Defines the overall Transformer architecture using custom components, outputting start and end logits for each token.
*   **`components/`**: Contains the building blocks:
    *   `attention.py`: `MultiHeadSelfAttention` layer.
    *   `positional_encoding.py`: `PositionalEncoding` layer.
    *   `transformer_block.py`: Combines attention and feed-forward layers into a standard `TransformerBlock`.
*   **`utils/evaluation.py`**: Contains helper functions for timing training and calculating SQuAD evaluation metrics (EM, F1 - requires integration into the training script for full use).

## Setup

1.  **Clone the repository.**
2.  **Create a Python environment** (e.g., using Conda or venv).
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    This will install all required packages with the exact versions used in development.
    
    Alternatively, you can install packages individually:
    ```bash
    pip install tensorflow numpy tokenizers transformers wandb datasets evaluate tqdm
    ```
    *(Ensure you have a version of TensorFlow compatible with your GPU and CUDA/cuDNN if using GPU acceleration).* 
4.  **Note on dataset:** The SQuAD dataset is automatically downloaded from Hugging Face when you run the script. No manual download is required.

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

---

## Project Log

This section documents the experiments, challenges, and findings during the development process.
**Note:** This log is not from the start of the project. 

### Initial Attempts (Custom Components)

*   **Goal:** Build a baseline QA model using custom implementations.
*   **Approach:**
    *   Used `keras.preprocessing.text.Tokenizer` (word-level).
    *   Implemented custom input formatting (`[CLS]`, `[SEP]`, padding).
    *   Used standard trainable `Embedding` layer and sinusoidal `PositionalEncoding`.
    *   Built a transformer encoder stack using custom `MultiHeadSelfAttention` and `FeedForward` layers.
    *   Trained on a small subset (1000 samples) initially, then the full dataset.
*   **Issues & Findings:**
    *   Word-level tokenizer limitations (OOV words, morphology).
    *   Padding necessity for batching highlighted.
    *   Positional encoding necessity explained (permutation invariance).
    *   Initial low hyperparameters led to poor performance (near-zero EM/F1).
    *   Training on the full dataset for 250 epochs yielded very low EM (0.0004) and F1 (0.0066), with many empty predictions. Model seemed unable to learn correct answer spans.
*   **Status:** Unsatisfactory performance.

### Hyperparameter Tuning & Refinements (Still Custom Focus)

*   **Goal:** Improve performance by adjusting hyperparameters and addressing potential issues.
*   **Approach:**
    *   Increased `VOCAB_SIZE`, `SEQUENCE_LEN`, `D_MODEL`, `NUM_HEADS`, `DFF`, `NUM_LAYERS`.
    *   Experimented with learning rate (fixed `1e-4`, then `5e-5`).
    *   Introduced learning rate scheduling (`PolynomialDecay`).
    *   Attempted to improve answer span finding in preprocessing (approximate matching, filtering invalid examples, character offset hints).
    *   Implemented early stopping based on validation loss (initially incorrect, monitored training loss later).
    *   Added attention masking to prevent attention to padding tokens.
*   **Issues & Findings:**
    *   Significant hyperparameter increases (up to 34M parameters) did not yield substantial improvements.
    *   Learning rate changes and scheduling didn't resolve the core issue.
    *   Improved preprocessing logic didn't lift scores significantly.
    *   Attention mask implementation was crucial but not sufficient.
    *   Performance remained near zero EM/F1. Suggests a fundamental problem.
*   **Status:** Still unsatisfactory. Fundamental learning issue suspected.

### Switching to Standard Components (Hugging Face Tokenizer)

*   **Goal:** Leverage battle-tested components to simplify the code and rule out custom implementation errors.
*   **Approach:**
    *   Replaced the custom tokenizer with `BertTokenizerFast` from Hugging Face (`bert-base-uncased`).
    *   Utilized the tokenizer's `offset_mapping` to find answer start/end token indices based on SQuAD character offsets. This replaced the fragile list-matching/approximate matching logic.
    *   Relied on the tokenizer's built-in attention mask and token type IDs.
*   **Issues & Findings:**
    *   Slight initial improvement observed in validation F1 after only 2 epochs compared to previous attempts (0.00567 F1 vs near zero), but still very low.
    *   Longer training (8.3 hours, 50 epochs) with large hyperparameters (`D_MODEL=768`, `NUM_LAYERS=6`) still resulted in EM=0 and F1=0.00513. Loss increased towards the end of training.
*   **Status:** Still fundamentally not working, despite using standard tokenizer and improved span finding.

### Addressing Potential Training Setup Issues

*   **Goal:** Correct potential flaws in the overall training pipeline and setup.
*   **Approach (Based on Gemini Suggestions):**
    *   Corrected dataset handling (shuffle/split *before* batching).
    *   Re-verified attention mask logic.
    *   Adjusted learning rate (increased initial LR) and added a warmup phase to the schedule.
    *   Improved monitoring (correct validation loss for early stopping) and added checkpointing.
    *   Attempted to enhance evaluation logic mapping predictions to answers.
*   **Issues & Findings:**
    *   Initial short run showed unstable validation loss (decreasing then increasing again). Final EM/F1 still zero after 16 epochs.
    *   Noticed high disk usage due to W&B artifact caching.
*   **Status:** Core learning problem persists.

### Current Strategy: Overfitting Sanity Check (Date: 2025-04-17)

*   **Goal:** Determine if the model can even memorize a tiny subset of the data. If not, it confirms a fundamental bug.
*   **Approach:** Train the current model setup on a very small number of examples (e.g., 32) for many epochs (200) with a constant learning rate (5e-5). Use the training data for validation and disable early stopping/checkpointing.
*   **Expected Outcome:** Training loss should approach zero, and accuracy/EM/F1 on the *training subset* should approach 100%.
*   **Actual Outcome:** The model failed to overfit. Training loss plateaued around 1.0-1.2, accuracies fluctuated around 60-85% and did not reach 100%. Final EM/F1 on the training subset was 0.0.
*   **Status:** Failed. Confirms a fundamental bug likely in data preprocessing (label generation), model output, or loss calculation.

### Next Steps: Debugging (Date: 2025-04-17)

*   **Goal:** Identify and fix the fundamental bug preventing the model from learning.
*   **Approach:** Systematically debug the pipeline, starting with the highest probability area:
    1.  **Data Preprocessing (`utils/hf_squad_preprocessing.py`):** Verify the logic mapping SQuAD character offsets to token start/end positions using the tokenizer's `offset_mapping` and `sequence_ids`. **FIXED: Corrected fallback logic for misaligned spans.**
    2.  **Model Output Layer (`models/qa_transformer.py`):** Check the final dense layers producing start/end logits.
    3.  **Loss Function Application (`train_squad.py`):** Confirm correct usage of `SparseCategoricalCrossentropy`.
    4.  **Attention Mask Usage:** Re-verify mask propagation and application.
*   **Status:** Debugging in progress. Preprocessing fix applied.

### Second Overfitting Sanity Check (Date: 2025-04-17)

*   **Goal:** Verify if the preprocessing fix allows the model to memorize a tiny subset.
*   **Approach:** Re-ran training on 32 examples for 200 epochs with fixed LR (5e-5), using training data for validation.
*   **Expected Outcome:** Training loss approaches zero, accuracy approaches 100%.
*   **Actual Outcome:** Success! Loss decreased significantly (to ~0.8-1.0), start/end token accuracies reached ~75-81%. The model demonstrated learning.
*   **Evaluation Bug:** The final EM/F1 calculation reported impossible scores (F1=14.21), indicating a separate bug in the evaluation script/function, not the training loop.
*   **Status:** Passed (Model is learning). Blocked on fixing evaluation logic.

### Next Steps: Fix Evaluation & Restore Config (Date: 2025-04-17)

*   **Goal:** Fix the EM/F1 calculation bug and restore the normal training configuration.
*   **Approach:**
    1.  Debug `utils/evaluation.py` (`compute_eval_metrics`) and its usage in `train_squad.py`. **FIXED: Corrected special token filtering in `get_predictions_from_logits` helper.**
    2.  Revert overfitting test changes in `train_squad.py` (samples, epochs, LR schedule, callbacks, validation set).
    3.  Run training on a larger dataset.
*   **Status:** Evaluation logic likely fixed, config restored for next test.

### Training Test 1 (Adam, 3e-5 LR, 3k Samples) (Date: 2025-04-17)

*   **Goal:** Test if the fixed preprocessing allows learning on a larger subset (3000 samples).
*   **Configuration:** `MAX_SAMPLES=3000`, `EPOCHS=30`, `BATCH_SIZE=16`, `SEQUENCE_LEN=384`, `D_MODEL=768`, `NUM_LAYERS=6`, `NUM_HEADS=12`, `DFF=2048`, `DROPOUT_RATE=0.1`, `Adam` optimizer, `INITIAL_LR=3e-5`, `END_LR=0.0`, `WarmupCosineDecay` schedule (10% warmup), EarlyStopping on `val_loss` (patience=5).
*   **Outcome:** Model started learning, `val_loss` decreased to 9.18 by epoch 7. However, learning stalled quickly, and early stopping triggered at epoch 12. Final metrics were very poor (EM=0, F1=3.08, scaled 0-100).
*   **Interpretation:** The preprocessing fix allowed initial learning, but the model configuration (architecture/hyperparameters) is not effective for this task/dataset size. It likely gets stuck in a poor local minimum early on.
*   **Status:** Learning confirmed, but performance unacceptable. Optimizer/LR tuning needed.

### Training Test 2 (AdamW, 3e-5 LR, 3k Samples) (Date: 2025-04-17)

*   **Goal:** Test AdamW optimizer with the previously best-performing learning rate.
*   **Approach:** Run experiment on 3000 samples using `AdamW` optimizer (weight decay=0.01) and `INITIAL_LR=3e-5`. Kept other settings the same as **Training Test 1**.
*   **Outcome:** Failed. User reported model did not learn effectively (similar to 1e-5 run).
*   **Interpretation:** AdamW with 3e-5 LR also failed to train the model effectively.
*   **Status:** Failed.

### Training Test 3 (AdamW, 5e-5 LR, 3k Samples) (Date: 2025-04-17)

*   **Goal:** Test AdamW optimizer with a higher learning rate.
*   **Approach:** Run experiment on 3000 samples using `AdamW` optimizer (weight decay=0.01) and `INITIAL_LR=5e-5`. Kept other settings the same as **Training Test 1**.
*   **Outcome:** Failed. Model did not learn. `val_loss` remained high (~11.49+) and did not improve after epoch 1. Early stopping triggered after 6 epochs. Final EM/F1 were 0.0.
*   **Interpretation:** AdamW with 5e-5 LR also failed. The optimizer/LR combination isn't the sole issue.
*   **Status:** Failed.

### Training Test 4 (Pre-LN, Adam, 3e-5 LR, 3k Samples) (Date: 2025-04-17)

*   **Goal:** Test if Pre-Layer Normalization improves training stability/performance.
*   **Approach:** Re-ran **Training Test 1** configuration (`Adam` optimizer, `INITIAL_LR=3e-5`, 3k samples, etc.) but with the modified `TransformerBlock` using Pre-LN.
*   **Outcome:** Improved stability. Model trained longer (16 epochs vs 10-12) and achieved a lower best `val_loss` (8.90 vs ~9.20). Training accuracies reached >10% (vs ~3%). However, `val_loss` still stalled, and final EM/F1 remained 0.0.
*   **Interpretation:** Pre-LN helped training stability compared to Post-LN, allowing the model to learn slightly better before getting stuck. The core performance issue remains.
*   **Status:** Partial success (stability improved), but overall performance still unacceptable.

### Training Test 5 (Pre-LN, Adam, 5e-5 LR, 3k Samples) (Date: 2025-04-17)

*   **Goal:** Test Pre-LN architecture with a slightly higher learning rate.
*   **Approach:** Ran experiment using the Pre-LN `TransformerBlock`, cleaned `MultiHeadSelfAttention`, `Adam` optimizer, `INITIAL_LR=5e-5`, and other settings from **Training Test 1** (3k samples, 30 epochs, etc.).
*   **Outcome:** Best result so far. `val_loss` decreased steadily to 8.77 (epoch 11). Training accuracies improved (>10%). However, `val_loss` stalled after epoch 11 due to overfitting on the 3k samples, triggering early stopping at epoch 16. Final EM/F1 remained 0.0.
*   **Interpretation:** Pre-LN and 5e-5 LR allow the model to learn more effectively than previous configurations, but it quickly overfits the small 3000-sample dataset, preventing it from reaching good performance. The model is learning, but needs either more data or is significantly less efficient than standard models.
*   **Status:** Learning improved, but overfitting on small dataset prevents meaningful QA performance.

### Next Steps: Establish Baseline Comparison (Date: 2025-04-17)

*   **Goal:** Determine if the poor performance is primarily due to the custom model/training limitations or insufficient data size (3k samples).
*   **Approach:** Fine-tune a standard pre-trained transformer model (e.g., `distilbert-base-uncased` or `bert-base-uncased`) on the **exact same 3000 training samples** using a standard library like Hugging Face `Trainer`. Compare its performance (EM/F1) after a short training period against the custom model's zero score.
*   **Rationale:**
    *   If the pre-trained baseline achieves significantly better scores (non-zero EM/F1) on the same small dataset, it indicates that the custom model implementation or its training setup is still inefficient or flawed compared to standard practices.
    *   If the pre-trained baseline *also* performs very poorly on the 3k samples, it strengthens the hypothesis that 3000 samples is simply insufficient data for this task, even for fine-tuning, suggesting that training the custom model on the full dataset is the necessary next step.
*   **Status:** Baseline implemented.

### TensorFlow 2.10 Compatible Baseline Results (Date: 2025-04-17)

*   **Goal:** Establish a strong baseline using pre-trained transformers to determine if our dataset size is sufficient.
*   **Approach:** Created a TensorFlow 2.10 compatible custom QA model using a pre-trained `distilbert-base-uncased` with a simple classification head for start/end token prediction. Trained on the same 3000 training samples for 3 epochs.
*   **Results:**
    *   **Epoch 1:** EM = 56.67%, F1 = 63.14%
    *   **Epoch 2:** EM = 62.50%, F1 = 68.90% 
    *   **Epoch 3:** EM = 63.50%, F1 = 69.44%
*   **Interpretation:** The pre-trained model achieves excellent performance even with just 3000 samples, clearly showing that the dataset size was not the limiting factor in our custom model's poor performance. Rather, our custom transformer implementation likely has flaws that prevent effective learning.
*   **Conclusions:**
    *   The strong baseline performance (~63% EM, ~69% F1) confirms that pre-trained models can be fine-tuned effectively with our current dataset size.
    *   Our custom model's inability to achieve non-zero EM/F1 scores indicates fundamental issues in the model architecture, training setup, or both.
    *   The difference highlights the value of using pre-trained models and following established architecture patterns.
*   **Next Steps:**
    *   Option 1: Continue improving the custom transformer, focusing on comparing its architecture more directly with proven implementations.
    *   Option 2: Use this pre-trained baseline for further experimentation, as it provides a strong starting point with minimal code complexity.
    *   Option 3: Try a hybrid approach - initialize parts of our custom model with pre-trained weights to get the best of both worlds.

### Custom Model Architecture Review (Date: 2025-04-17)

*   **Goal:** Review the core components of the custom transformer implementation for potential issues.
*   **Components Reviewed:** `components/positional_encoding.py`, `components/attention.py`, `components/transformer_block.py`, `models/qa_transformer.py`.
*   **Findings:**
    *   Positional Encoding, Attention (using Keras MHA), and the Pre-LN Transformer Block implementations appear standard and correct.
    *   The main architectural deviation is in the **output head** (`models/qa_transformer.py`). It uses **two separate `Dense(1)` layers** to predict start and end logits independently. The more standard approach (used in BERT, DistilBERT, and the successful baseline) is a **single shared `Dense(2)` layer**, followed by splitting the output. This difference might make learning the relationship between start and end tokens more difficult.
*   **Status:** Review complete. One potential area for architectural improvement identified (output head). Proceeding to simplify training configuration first.

### Training Test 6 (Pre-LN, AdamW, Fixed LR, 3k Samples) (Date: 2025-04-17)

*   **Goal:** Test the custom model with a simplified optimizer setup, mimicking the baseline (AdamW used for TF 2.10 compatibility with weight decay).
*   **Configuration:** Pre-LN blocks, `tfa.optimizers.AdamW`, fixed `LEARNING_RATE=5e-5`, `WEIGHT_DECAY=0.01`, 3k samples, 30 epochs.
*   **Outcome:** Failed. Model did not learn. Loss remained high (~11), accuracies near zero (~1-2%). Early stopping triggered after 6 epochs. Final F1 (subset) ~5.3.
*   **Interpretation:** Simplifying the optimizer configuration did not resolve the core learning issue. This reinforces the suspicion that architectural differences, particularly the output head, might be hindering performance.
*   **Status:** Failed. Proceeding with architectural changes.

### Next Steps: Modify Custom Model Output Head (Date: 2025-04-17)

*   **Goal:** Align the custom model's output head with standard practice and the successful baseline.
*   **Approach:** Modify `models/qa_transformer.py` to use a **single shared `Dense(2)` layer** for predicting start/end logits, followed by splitting the output, instead of the current two separate `Dense(1)` layers.
*   **Rationale:** This may help the model learn the relationship between start and end tokens more effectively.
*   **Status:** Change implemented, proceeding to test.

### Training Test 7 (Pre-LN, AdamW, Dense(2) Output, 3k Samples) (Date: 2025-04-17)

*   **Goal:** Test the custom model with the standard Dense(2) output head.
*   **Configuration:** Pre-LN blocks, `tfa.optimizers.AdamW`, fixed `LEARNING_RATE=5e-5`, `WEIGHT_DECAY=0.01`, **Dense(2) output head**, 3k samples, 30 epochs.
*   **Outcome:** Failed. Performance was identical to Training Test 6. Loss remained high (~11), accuracies near zero (~1-2%). Early stopping triggered after 6 epochs. Final F1 (subset) ~2.8.
*   **Interpretation:** Changing the output head architecture did not resolve the fundamental learning issue. The problem likely lies elsewhere in the model or its configuration/initialization.
*   **Status:** Failed.

### Next Steps: Sanity Check & Transfer Learning (Date: 2025-04-17)

Given the persistent issues with the custom model trained from scratch, even after aligning its optimizer and output head with the baseline, two main paths remain viable:

1.  **Hyperparameter Simplification Sanity Check (Option A):** Radically reduce the model size (`NUM_LAYERS`, `D_MODEL`, etc.) and train briefly. If this tiny model shows *any* sign of learning (decreasing loss, increasing accuracy beyond random chance), it suggests the larger configuration might be unstable or require different hyperparameters/initialization. If the tiny model *also* fails completely, it strengthens the suspicion of a subtle bug in the custom layer implementations or their interactions.
2.  **Transfer Learning / Using Pre-trained Weights (Option B):** This is a very common and effective technique in NLP. Instead of initializing all model weights randomly, we load weights from a model that has already been trained on a massive text corpus (like `bert-base-uncased` or `distilbert-base-uncased`).
    *   **How it Works:** We would modify `models/qa_transformer.py` to load the corresponding pre-trained weights into our custom layers. For example, the weights for the `Embedding` layer and the weights for the attention and feed-forward layers within each `TransformerBlock` can often be mapped directly if the dimensions match. The output head (`Dense(2)`) would typically still be initialized randomly as it's specific to our QA task.
    *   **Why do this?**
        *   **Better Initialization:** Pre-trained weights provide a much better starting point than random initialization. The model already understands language structure, syntax, and some semantics.
        *   **Faster Convergence:** Fine-tuning usually requires fewer epochs and less data to reach good performance compared to training from scratch.
        *   **Higher Potential Performance:** Leveraging the knowledge learned from vast amounts of text often leads to better generalization and higher final scores.
        *   **Debugging Aid:** If the model *still* fails even with pre-trained weights loaded into the core components, it very strongly points to a bug in the remaining parts (e.g., how the components are connected, mask handling, or the output layer).
*   **Decision:** Proceeding with Option A (Hyperparameter Simplification) first as a final quick check before potentially moving to Option B (Transfer Learning).

### Training Test 8 (Simplified Hyperparameters) (Date: 2025-04-17)

*   **Goal:** Sanity check if a drastically smaller custom model shows any learning.
*   **Configuration:** Pre-LN, AdamW, Dense(2) Output, `NUM_LAYERS=2`, `D_MODEL=256`, `NUM_HEADS=4`, `DFF=1024`, 3k samples, `EPOCHS=5`.
*   **Outcome:** Failed. Model showed no signs of learning. Loss remained high (~11.6), accuracies near zero (~1%). Final F1 (subset) ~3.5.
*   **Interpretation:** The failure of even a minimal custom model strongly suggests a subtle bug in the component implementations/interactions, or that training this architecture from scratch on this data scale is infeasible without better initialization.
*   **Status:** Failed. Simple configuration does not learn.

### Next Steps: Transfer Learning (Date: 2025-04-17)

*   **Goal:** Leverage pre-trained weights to initialize the custom model structure, bypassing issues with training from scratch.
*   **Approach:** Modify `models/qa_transformer.py` to load weights from a pre-trained model (e.g., `distilbert-base-uncased`) into the `Embedding` layer and the `TransformerBlock`s. The final `qa_outputs` head will remain randomly initialized.
*   **Rationale:** This provides a strong initialization, making training feasible and allowing us to verify if the custom *structure* can work when properly initialized.
*   **Status:** Proceeding with implementation.

### Debugging & Baseline Efforts (Date: 2025-04-17)

*   **Goal:** After previous attempts failed to yield meaningful results with the custom transformer, the focus shifted to diagnosing the root cause and establishing a reliable performance baseline.
*   **Steps Taken & Findings:**
    1.  **TF 2.10 Compatibility Issues:** Encountered multiple compatibility errors (`AttributeError`, `ImportError`, `Reduction` errors) when trying to run a standard Hugging Face fine-tuning script (`baseline_hf_trainer.py`) with TF 2.10. These were eventually resolved by creating a simplified baseline script (`simpler_baseline_tf210.py`) using a custom Keras model structure wrapping a pre-trained DistilBERT and a manual training loop.
    2.  **Baseline Success:** The TF 2.10-compatible baseline script successfully trained on the **same 3000 samples** used for the custom model tests. It achieved significant performance after only 3 epochs (**EM ~63.5%, F1 ~69.4%**).
        *   **Achievement:** This definitively proved that the **dataset size (3k samples) is sufficient** for fine-tuning a standard transformer for this task and that the **preprocessing/evaluation logic is sound.**
        *   **Conclusion:** The poor performance of the custom model stems from issues within its **architecture, implementation, or training from scratch**, not the data or basic setup.
    3.  **Custom Model Re-evaluation:** Returned to the custom model (`train_squad.py`) with the knowledge that the task *should* be learnable.
        *   **Architecture Review:** Reviewed `PositionalEncoding`, `Attention`, `TransformerBlock` (Pre-LN), and `qa_transformer` model definition. Confirmed most components seemed standard, but identified the separate `Dense(1)` output layers as a deviation from the baseline's shared `Dense(2)`.
        *   **Test 6 (Simplified Optimizer):** Mimicked baseline optimizer settings (AdamW for TF 2.10 compatibility, fixed LR, weight decay). **Result: Failed.** No learning improvement.
        *   **Test 7 (Standard Output Head):** Changed the custom model's output to a shared `Dense(2)` layer, matching the baseline. **Result: Failed.** No learning improvement.
        *   **Test 8 (Hyperparameter Simplification):** Drastically reduced model size (`layers=2`, `d_model=256`, etc.) for a sanity check. **Result: Failed.** Even the tiny model showed no signs of learning.
        *   **Test 9 & 10 (Transfer Learning Attempts):** Modified the custom model (`models/qa_transformer.py`) to load pre-trained weights from `distilbert-base-uncased`.
            *   **Attempt 1:** Failed to load weights due to incorrect layer access (`AttributeError: 'TFDistilBertMainLayer' object has no attribute 'get_layer'`).
            *   **Attempt 2:** Corrected layer access. Successfully loaded **Embedding**, **FFN**, and **LayerNorm** weights. However, **failed to load Multi-Head Attention weights** due to shape/structure incompatibility between the standard Keras `MultiHeadAttention` layer and the internal layers of the HF `TFDistilBertModel`.
            *   **Result:** Because attention weights weren't loaded, the model still trained attention from scratch and **failed to learn effectively** (Loss ~11, Acc ~1-2%, F1 ~4%).
*   **Overall Achievements Today:**
    *   Successfully created a **working, high-performing baseline** model for TF 2.10.
    *   **Pinpointed the problem** to the custom model implementation/training-from-scratch difficulties.
    *   **Ruled out data sufficiency** as the primary issue.
    *   Gained experience with **debugging TF compatibility** issues.
    *   Partially implemented **transfer learning** for the custom architecture.
*   **Reason for Custom Model Failure:** Despite extensive debugging and aligning components (optimizer, output head) with the successful baseline, the custom model consistently failed to learn. The final attempt showed that even loading *some* pre-trained weights wasn't enough when the crucial attention weights couldn't be transferred due to layer incompatibilities. This highlights the significant challenge of training complex models like transformers from scratch without proper initialization (like pre-trained weights) and potentially reveals subtle implementation issues or the need for very specific hyperparameter tuning beyond standard defaults.
*   **Current Status:** The **custom model is still not functional**. The **baseline model (`simpler_baseline_tf210.py`) is functional and performs well.** The attempt to use transfer learning on the custom model structure was informative but ultimately unsuccessful due to MHA weight incompatibility.

### Reverting to Custom Model (Date: 2025-04-18)

*   **Goal:** Return to the custom model architecture and training setup from before the transfer learning attempts, specifically aiming to replicate the conditions of **Training Test 5** (Pre-LN, Adam, 5e-5 LR, Dense(2) output, Keras MHA).
*   **Rationale:** Although previous custom model tests failed to achieve good performance, the user expressed a preference for working with the custom implementation. This step reverts the code to the most promising state achieved with the custom model.
*   **Changes Made:**
    *   Removed weight loading logic from `models/qa_transformer.py`.
    *   Reverted hyperparameters in `train_squad.py` (`bert-base-uncased` tokenizer, `D_MODEL=768`, `LAYERS=6`, `HEADS=12`, `DFF=2048`, `EPOCHS=30`, `MAX_SAMPLES=3000`).
    *   Switched optimizer back to `tf.keras.optimizers.Adam`.
    *   Re-implemented `PolynomialDecay` learning rate schedule with warmup (Initial LR `5e-5`).
    *   Removed `tensorflow-addons` dependency.
    *   Updated early stopping patience to 5 and checkpoint filename.
*   **Status:** Code reverted. Ready to run the custom model configuration again (effectively **Training Test 11**, replicating Test 5 conditions).

### Training Test 11 (Reverted Custom Model - 3k Samples) (Date: 2025-04-18)

*   **Goal:** Re-run the custom model configuration from Test 5 after reverting the transfer learning changes to confirm baseline behavior.
*   **Configuration:** Pre-LN blocks, `bert-base-uncased` tokenizer, `Adam` optimizer, `PolynomialDecay` LR schedule (`5e-5` initial), `MAX_SAMPLES=3000`, `EPOCHS=30`, EarlyStopping patience=5.
*   **Outcome:** Training ran for 10 epochs before early stopping triggered. Best `val_loss` was ~8.82. Final metrics on validation subset (100 samples): **EM = 4.0%, F1 = 6.94%**.
*   **Interpretation:** Confirmed the reverted model learns slightly but performs poorly and stalls/overfits quickly on the small 3k dataset, achieving marginal non-zero scores.
*   **Status:** Completed. Performance remains very low.

### Training Test 12 (Reverted Custom Model - 87k Samples) (Date: 2025-04-18)

*   **Goal:** Test the reverted custom model configuration with a larger dataset to see if performance improves.
*   **Configuration:** Same as Test 11, but with `MAX_SAMPLES=87000` (using full training set) and explicitly ran for `EPOCHS=3`.
*   **Outcome:** Training completed 3 epochs. Best `val_loss` was ~8.52 (slight improvement over Test 11). Final metrics on validation subset (100 samples): **EM = 4.0%, F1 = 7.03%**.
*   **Interpretation:** Increasing the dataset size from 3k to 87k allowed the model to train for the specified 3 epochs without immediate overfitting and achieved slightly lower validation loss. However, the end performance metrics (EM/F1 on the subset) remained essentially unchanged and extremely low, indicating the model still struggles fundamentally even with more data.
*   **Status:** Completed. Minimal improvement observed despite more data.

### Training Test 13 (Reverted Custom Model - 10k Samples) (Date: 2025-04-18)

*   **Goal:** Test the reverted custom model configuration with an intermediate dataset size (10k samples) and allow for longer training.
*   **Configuration:** Same as Test 11/12, but with `MAX_SAMPLES=10000` and `EPOCHS=20` (EarlyStopping patience=5).
*   **Outcome:** Training ran for 9 epochs before early stopping triggered. Best `val_loss` was ~8.68 (achieved at epoch 4). `val_loss` increased after epoch 4, indicating overfitting. Final metrics on validation subset (100 samples) based on weights from epoch 4: **EM = 5.0%, F1 = 6.35%**.
*   **Interpretation:** The model clearly showed signs of learning on the 10k samples, achieving its best validation score around epoch 4 before overfitting. The EM score was slightly higher than the 87k run (Test 12), while F1 was slightly lower, but both runs show very poor absolute performance. The overfitting pattern suggests the model *can* learn from this data but lacks the capacity or regularization to generalize well, even on 10k samples, when trained from scratch.
*   **Status:** Completed. Confirms model learns but overfits quickly; performance remains very low.

### Training Test 14 (Dropout 0.2 - 10k Samples) (Date: 2025-04-18)

*   **Goal:** Test if increasing dropout (0.2) improves generalization on the 10k sample dataset.
*   **Configuration:** Same as Test 13, but with `DROPOUT_RATE=0.2`.
*   **Outcome:** Training ran for 9 epochs (early stopping). Best `val_loss` ~8.76 (epoch 4), worse than Test 13. Final metrics on validation subset (100 samples): **EM = 0.0%, F1 = 2.79%**.
*   **Interpretation:** Increasing dropout to 0.2 seemed to hinder learning or require more training time, resulting in worse performance than 0.1 dropout on this dataset size.
*   **Status:** Completed. Higher dropout was detrimental in this setup.

### Training Test 15 (Dropout 0.1 - Full Dataset - 10 Epochs) (Date: 2025-04-18)

*   **Goal:** Train the model with 0.1 dropout on the full SQuAD training set for 10 epochs.
*   **Configuration:** Reverted `DROPOUT_RATE=0.1`, `MAX_SAMPLES=87000` (full dataset), `EPOCHS=10`.
*   **Outcome:** Training completed all 10 epochs. `val_loss` steadily decreased throughout, reaching a best of **~8.39** (epoch 9/10). Final metrics on validation subset (100 samples): **EM = 5.0%, F1 = 9.70%**. Runtime: **~7774 seconds (~2.17 hours)**.
*   **Interpretation:** Best validation loss and F1 score achieved so far with the custom model. Training on the full dataset allowed consistent improvement over 10 epochs without signs of overfitting yet. Suggests the model might benefit from continued training.
*   **Status:** Completed. Best custom model results so far.

### Training Test 16 (Dropout 0.1 - Full Dataset - 20 Epochs) (Date: 2025-04-18)

*   **Goal:** Continue training the model on the full dataset for longer (20 epochs) to see if performance continues to improve.
*   **Configuration:** Same as Test 15: `DROPOUT_RATE=0.1`, `MAX_SAMPLES=87000` (full dataset), but `EPOCHS=20`.
*   **Outcome:** Training completed all 20 epochs. `val_loss` continued to decrease fairly consistently, reaching a best of **~8.03** (epoch 19). Final metrics on validation subset (100 samples): **EM = 8.0%, F1 = 15.95%**. Runtime: **~15717 seconds (~4.37 hours)**.
*   **Interpretation:** This run yielded the **best validation loss and EM/F1 scores so far** for the custom model trained from scratch. The model demonstrated continued learning over 20 epochs on the full dataset without severe overfitting. This confirms the architecture's capability to learn the task, albeit much slower and less effectively than the pre-trained baseline. The slow but steady improvement suggests further training *could* yield slightly better results, but likely with diminishing returns.
*   **Status:** Completed. Best custom model performance achieved.

### Training Test 17 (Dropout 0.15, AdamW, WD 5e-5 - 10k Samples) (Date: 2025-04-19)

*   **Goal:** Test if adding slight weight decay (`5e-5`) and increasing dropout (`0.15`) improves generalization on the 10k dataset.
*   **Configuration:** Same as Test 13 (10k samples, `EPOCHS=20`), but with `DROPOUT_RATE=0.15`, `AdamW` optimizer, and `WEIGHT_DECAY=5e-5`.
*   **Outcome:** Training ran for 9 epochs before early stopping triggered. Best `val_loss` was ~8.78 (epoch 4), similar to Test 13 (~8.68), but slightly worse. Final metrics on validation subset (100 samples) based on weights from epoch 4: **EM = 3.0%, F1 = 5.8%**.
*   **Interpretation:** Adding slight weight decay and increasing dropout to 0.15 didn't significantly hinder the model's ability to learn on 10k samples (overfitting still occurred around epoch 4), but the peak performance was slightly lower than with 0.1 dropout and no weight decay (Test 13). It remains to be seen if this combination aids generalization on the full dataset.
*   **Status:** Completed. Overfitting pattern similar to Test 13, slightly worse peak metrics.

### Training Test 18 (Warmup, Dropout 0.15, AdamW, WD 5e-5 - 10k Samples) (Date: 2025-04-19)

*   **Goal:** Test if adding an explicit linear warmup phase to the learning rate schedule (AdamW, 5e-5 initial LR, 0.15 dropout, 5e-5 WD) stabilizes initial training and improves peak performance before overfitting on the 10k dataset.
*   **Rationale:** Warmup gradually increases the learning rate at the start of training, preventing potentially large and unstable updates when the model weights are still random, which is a standard practice for training Transformers.
*   **Configuration:** Same as Test 17, but implemented a custom `WarmupPolynomialDecay` schedule.
*   **Outcome:** Training ran for 8 epochs before early stopping. Best `val_loss` was ~8.77 (epoch 3), reached slightly earlier than Test 17. Final metrics on validation subset (100 samples) based on weights from epoch 3: **EM = 3.0%, F1 = 7.7%**.
*   **Interpretation:** The explicit warmup slightly stabilized initial training and led to a marginally better F1 score compared to Test 17, although the best `val_loss` and overfitting pattern remained similar. The warmup didn't prevent overfitting on the small dataset but might be beneficial for stability in longer runs on the full dataset.
*   **Status:** Completed. Warmup provided slight stability/F1 improvement but didn't fundamentally change the 10k overfitting behavior.

### Training Test 19 (Label Smoothing Attempt) (Date: 2025-04-19)

*   **Goal:** Test if adding label smoothing (0.1) could improve regularization.
*   **Configuration:** Based on Test 18 (10k samples, LR=2e-5, WD=5e-5, Dropout=0.15, AdamW, Warmup).
*   **Outcome:** Failed immediately with `TypeError` because `tf.keras.losses.sparse_categorical_crossentropy` does not accept `label_smoothing` in TF 2.10, and neither does the class constructor `tf.keras.losses.SparseCategoricalCrossentropy`.
*   **Interpretation:** Label smoothing is not readily available via the standard loss functions in TF 2.10. Reverted the change.
*   **Status:** Failed due to TF incompatibility.

### Training Test 20 (12 Layers, Dropout 0.1 - 10k Samples) (Date: 2025-04-19)

*   **Goal:** Test if increasing model depth (12 layers) improves performance on the 10k subset, using 0.1 dropout.
*   **Configuration:** `NUM_LAYERS=12`, `DROPOUT_RATE=0.1`, 10k samples, LR=2e-5, WD=5e-5, AdamW, Warmup.
*   **Outcome:** Training ran for 11 epochs (early stopping). Best `val_loss` ~8.67 (epoch 6). Final metrics on validation subset (100 samples) based on weights from epoch 6: **EM = 6.0%, F1 = 9.1%**.
*   **Interpretation:** Best EM/F1 scores achieved so far on the 10k subset. Increasing depth to 12 layers with 0.1 dropout allowed the model to reach slightly better peak performance before overfitting, compared to the 6-layer model.
*   **Status:** Completed. Best 10k subset performance so far.

### Training Test 21 (12 Layers, Dropout 0.05 - 10k Samples) (Date: 2025-04-19)

*   **Goal:** Test if reducing dropout (0.05) further improves the 12-layer model on the 10k subset.
*   **Configuration:** `NUM_LAYERS=12`, `DROPOUT_RATE=0.05`, 10k samples, LR=2e-5, WD=5e-5, AdamW, Warmup.
*   **Outcome:** Training ran for 10 epochs (early stopping). Best `val_loss` ~8.67 (epoch 5). Final metrics on validation subset (100 samples) based on weights from epoch 5: **EM = 2.0%, F1 = 4.7%**.
*   **Interpretation:** Reducing dropout to 0.05 significantly worsened performance compared to 0.1 dropout for the 12-layer model on this dataset size, likely due to increased overfitting. Confirms 0.1 dropout is preferable for this setup.
*   **Status:** Completed. Lower dropout was detrimental.

### Training Test 22 (12 Layers, Dropout 0.05 - Full Dataset - 12 Epochs) (Date: 2025-04-19)

*   **Goal:** Test the 12-layer model on the full dataset, using the low dropout rate (0.05) that performed poorly on the 10k subset.
*   **Configuration:** `NUM_LAYERS=12`, `DROPOUT_RATE=0.05`, `MAX_SAMPLES=None`, `EPOCHS=12`, LR=2e-5, WD=5e-5, AdamW, Warmup.
*   **Outcome:** Training ran for 9 epochs before early stopping triggered. Best `val_loss` was ~8.63 (achieved at epoch 4). Final metrics on validation subset (100 samples) based on weights from epoch 4: **EM = 1.0%, F1 = 5.0%**. Runtime: **~13316 seconds (~3.7 hours)**.
*   **Interpretation:** Performance was significantly worse than the 6-layer model run (Test 16). The low dropout rate (0.05) likely led to rapid overfitting on the full dataset with the deeper 12-layer model, causing validation loss to stop improving very early. This configuration is not effective.
*   **Status:** Completed. Performance much worse than Test 16, confirms 0.05 dropout is detrimental for this deeper model.

### Training Test 23 (12 Layers, Dropout 0.15 - Full Dataset - 20 Epochs) (Date: 2025-04-20)

*   **Goal:** Test the 12-layer model on the full dataset with potentially better regularization (0.15 dropout, 5e-5 WD) and higher LR (5e-5).
*   **Configuration:** `NUM_LAYERS=12`, `DROPOUT_RATE=0.15`, `MAX_SAMPLES=None`, `EPOCHS=20`, `INITIAL_LR=5e-5`, `WEIGHT_DECAY=5e-5`, AdamW, Warmup.
*   **Outcome:** Training ran for 8 epochs before early stopping triggered. Best `val_loss` was ~8.60 (achieved at epoch 3). Final metrics on validation subset (100 samples) based on weights from epoch 3: **EM = 5.0%, F1 = 8.3%**. Runtime: **~12354 seconds (~3.4 hours)**.
*   **Interpretation:** The 12-layer model again showed signs of rapid overfitting on the full dataset, achieving peak performance at epoch 3 before validation loss increased. The results were better than Test 22 (0.05 dropout) but still substantially worse than the best 6-layer run (Test 16). This suggests the 12-layer model is more difficult to optimize/regularize effectively when trained from scratch.
*   **Status:** Completed. Performance still well below the 6-layer peak.

### Training Test 24 (6 Layers, Dropout 0.1, AdamW, WD 5e-5 - Full Dataset) (Date: 2025-04-20)

*   **Goal:** Re-test the 6-layer configuration (like Test 16) but using the AdamW optimizer and slight weight decay used in recent tests.
*   **Configuration:** `NUM_LAYERS=6`, `DROPOUT_RATE=0.1`, `MAX_SAMPLES=None`, `EPOCHS=50`, `INITIAL_LR=5e-5`, `WEIGHT_DECAY=5e-5`, AdamW, Warmup.
*   **Outcome:** Training ran for 8 epochs before early stopping triggered. Best `val_loss` was ~8.62 (achieved at epoch 3). Final metrics on validation subset (100 samples) based on weights from epoch 3: **EM = 5.0%, F1 = 6.5%**. Runtime: **~6386 seconds (~1.8 hours)**.
*   **Interpretation:** Performance was substantially worse than Test 16 (EM=8%/F1=16%), despite using the same layer count and dropout rate. This highlights that standard `Adam` (used in Test 16) was more effective for this specific model/task than `AdamW` with minimal weight decay. Overfitting occurred quickly.
*   **Status:** Completed. Performance well below Test 16, suggesting AdamW+WD is less effective here.

### Training Test 25 (Replicating Test 16 w/ Warmup - Full Dataset) (Date: 2025-04-21)

*   **Goal:** Replicate and potentially extend the successful Test 16 configuration (6 Layers, 0.1 Dropout, Adam optimizer, 5e-5 Initial LR, Full Dataset) using the correctly implemented warmup schedule.
*   **Configuration:** `NUM_LAYERS=6`, `DROPOUT_RATE=0.1`, `MAX_SAMPLES=None`, `EPOCHS=40`, `INITIAL_LR=5e-5`, standard `Adam`, `WarmupPolynomialDecay`. Checkpointing enabled.
*   **Outcome:** Training ran for a total of 16 epochs across two sessions (initial run stopped, resumed from checkpoint) before early stopping triggered. Best `val_loss` was **~8.28** (achieved at epoch 6 of the *resumed* session, equivalent to epoch 11 overall). Final metrics based on weights from this best epoch: **EM = 7.0%, F1 = 12.39%**. Total runtime ~5.5 hours.
*   **Interpretation:** Performance plateaued similarly to Test 16, hitting the best validation loss around epoch 11. The final metrics were slightly worse than the peak achieved in Test 16 (8.0/16.0). Confirms the performance ceiling for this architecture trained from scratch. The explicit warmup didn't yield further gains.
*   **Status:** Completed. Performance ceiling confirmed, slightly below Test 16 peak.
Visit the W&B dashboard linked in your terminal to monitor your training runs in real-time. 
