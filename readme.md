# Fine-Tune Ministral-8B on Custom Medical Data (macOS, MLX, Hugging Face)

This project demonstrates how to fine-tune the Ministral-8B LLM using custom medical symptom-diagnosis data on a Mac with MLX and Hugging Face libraries.

## Prerequisites
- macOS with Python 3.x
- [MLX](https://github.com/ml-explore/mlx) installed
- [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/quick-start)

## Setup

1. **Install dependencies:**
   ```sh
   pip install huggingface-hub transformers numpy mlx
   pip install mlx-lm --no-dependencies
   pip install -r requirements.txt
   ```

2. **Prepare the dataset:**
   - Run the data creation script:
     ```sh
     python data_create.py
     ```
   - Ensure you have a folder named `data` containing `train.jsonl`, `test.jsonl`, and `valid.jsonl`.

3. **Login to Hugging Face:**
   ```sh
   huggingface-cli login
   ```
   - Provide your Hugging Face token when prompted.

4. **Download the base model:**
   ```sh
   huggingface-cli download mlx-community/Ministral-8B-Instruct-2410-4bit
   ```

## Training

Fine-tune the model using LoRA:
```sh
python -m mlx_lm.lora \
    --model mlx-community/Ministral-8B-Instruct-2410-4bit \
    --data data \
    --train \
    --fine-tune-type lora \
    --batch-size 4 \
    --num-layers 16 \
    --iters 1000 \
    --adapter-path adapters
```

## Testing

Test with batch-size 2
```sh
python -m mlx_lm.lora \
    --model mlx-community/Ministral-8B-Instruct-2410-4bit \
    --adapter-path adapters \
    --data data \
    --batch-size 2 \
    --test
```

Test the trained LLM adapters:
```sh
python -m mlx_lm.generate \
    --model mlx-community/Ministral-8B-Instruct-2410-4bit \
    --max-tokens 500 \
    --adapter-path adapters \
    --prompt "Symptoms: I have been experiencing memory loss, stiffness and difficulty walking. Question: What could be the diagnosis I have?"
```

## Save the Fine-Tuned Model
```sh
python -m mlx_lm.fuse \
    --model mlx-community/Ministral-8B-Instruct-2410-4bit \
    --adapter-path adapters \
    --save-path model/fine-tuned_Ministral-8B-custom-dataset
```

## Test the Saved Model
```sh
python -m mlx_lm.generate \
    --model model/fine-tuned_Ministral-8B-custom-dataset \
    --max-tokens 500 \
    --prompt "Symptoms: I have been experiencing memory loss, stiffness and difficulty walking. Question: What could be the diagnosis I have?"
```

## Upload to Hugging Face
Follow Hugging Face documentation to upload your fine-tuned model.
Example
huggingface-cli upload model/fine-tuned_Ministral-8B-custom-dataset 
---

## Sample Fine Tuned Model
https://huggingface.co/Somasish01/fine-tuned-ministral-8b-custom-data

### About `data_create.py`
This script prepares and splits your dataset for fine-tuning. It loads a CSV of symptom-diagnosis pairs, reformats them into prompt-response text, shuffles and splits the data, and saves them as JSONL files for training, testing, and validation.

---

For more details, see comments in `data_create.py`.

## Parameter Explanations

### Training (`mlx_lm.lora`)
- `--model`: Path or Hugging Face ID of the base model to fine-tune.
- `--data`: Directory containing `train.jsonl`, `test.jsonl`, and `valid.jsonl` files.
- `--train`: Flag to enable training mode.
- `--fine-tune-type`: Type of fine-tuning; `lora` uses Low-Rank Adaptation for efficient training.
- `--batch-size`: Number of samples processed in each training batch.
- `--num-layers`: Number of model layers to fine-tune (higher values may require more memory).
- `--iters`: Number of training iterations (steps).
- `--adapter-path`: Directory to save the LoRA adapter weights after training.

### Testing (`mlx_lm.lora` with `--test`)
- `--model`: Path or Hugging Face ID of the base model.
- `--adapter-path`: Path to the trained LoRA adapters.
- `--data`: Directory containing the test data (`test.jsonl`).
- `--batch-size`: Number of samples processed in each test batch.
- `--test`: Flag to enable test mode (evaluates the model on the test set).

### Save the Fine-Tuned Model (`mlx_lm.fuse`)
- `--model`: Path or Hugging Face ID of the base model.
- `--adapter-path`: Path to the trained LoRA adapters.
- `--save-path`: Output directory to save the fully fused, fine-tuned model.

### Test the Saved Model (`mlx_lm.generate`)
- `--model`: Path to the saved, fine-tuned model.
- `--max-tokens`: Maximum number of tokens to generate in the output.
- `--prompt`: Input prompt for the model to generate a response.
- `--adapter-path` (optional): Path to adapters if using LoRA adapters (not needed if using fused model).

---
