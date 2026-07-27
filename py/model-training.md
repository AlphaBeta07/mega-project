# Model Training — Complete Documentation

This document covers everything about how the AI model was built, what parameters were used, and the full step-by-step training process.

---

## The Two Models & How They Connect

The app uses two completely separate AI models chained together like a production line:

```
Your Audio/Video
      │
      ▼
┌─────────────────────────────┐
│  MODEL 1: Faster-Whisper    │  ← Listens & converts speech → text
│  (Speech-to-Text)           │
└──────────────┬──────────────┘
               │  raw transcript (plain text)
               ▼
┌─────────────────────────────┐
│  MODEL 2: Phi-3 Mini 3.8B   │  ← Reads text & writes structured notes
│  (Your fine-tuned LLM)      │
└──────────────┬──────────────┘
               │
               ▼
     Structured Markdown Notes
```

---

## Model 1 — Faster-Whisper (Speech-to-Text)

This was **not trained** — it is a pre-built model that was loaded and used directly.

```python
from faster_whisper import WhisperModel

def load_stt_model():
    device       = "cuda"    # uses your RTX 4050 GPU
    compute_type = "float16" # fast half-precision on GPU
    return WhisperModel("small", device=device, compute_type=compute_type)
```

- Library: `faster-whisper` (4x faster re-implementation of OpenAI Whisper)
- Model size: **small** — 244M parameters
- Runs entirely on local GPU, zero internet needed
- Converts any audio/video into raw text transcript

---

## Model 2 — Phi-3 Mini 3.8B (Language Model)

This is the core brain of the application, custom-built through 3 stages.

### What is Phi-3 Mini?

**Microsoft Phi-3 Mini** (`Phi-3-mini-4k-instruct`) is a **3.8 billion parameter** language model developed by Microsoft Research. Despite being small, it rivals models 3–5x its size in reasoning and instruction-following.

- Architecture type: `MistralForCausalLM` (decoder-only Transformer)
- Context window: 4,096 tokens
- License: MIT (open source, commercial use allowed)

### Exact Architecture Parameters (from `config.json`)

| Parameter | Value | What it means |
|-----------|-------|----------------|
| `hidden_size` | 3072 | Width of each layer — size of every token's embedding vector |
| `intermediate_size` | 8192 | Inner size of the Feed-Forward Network (FFN) inside each Transformer block |
| `num_hidden_layers` | 32 | Total number of stacked Transformer blocks |
| `num_attention_heads` | 32 | Number of parallel attention heads per layer |
| `num_key_value_heads` | 32 | Used for Grouped Query Attention |
| `head_dim` | 96 | Dimension per attention head (3072 / 32 = 96) |
| `max_position_embeddings` | 4096 | Maximum sequence/context length |
| `sliding_window` | 2048 | Efficient sliding window attention size |
| `vocab_size` | 32,064 | Number of unique tokens the model knows |
| `hidden_act` | silu | SiLU activation function |
| `rms_norm_eps` | 1e-05 | Numerical stability for RMSNorm layers |
| `rope_theta` | 10000.0 | Base frequency for Rotary Positional Embeddings |
| `torch_dtype` | bfloat16 | 16-bit Brain Float precision |
| `attention_dropout` | 0.0 | No attention dropout |

### How it is 3.8 Billion Parameters

```
Embedding Layer:
   vocab_size × hidden_size = 32,064 × 3,072 ≈ 98M params

× 32 Transformer Layers, each containing:
   Self-Attention:
        Q, K, V, O projections = 4 × (3072 × 3072) ≈ 37.7M
   Feed-Forward Network (FFN):
        gate_proj + up_proj + down_proj = 3 × (3072 × 8192) ≈ 75.5M
   RMSNorm (2 per layer, negligible)

Output Head (LM Head):
   hidden_size × vocab_size ≈ 98M params

Total ≈ 98M + 32 × (37.7M + 75.5M) + 98M
      ≈ 98M + 32 × 113.2M + 98M
      ≈ 98M + 3,622M + 98M
      ≈ 3,818M ≈ 3.8 Billion Parameters
```

---

## Training Parameters

### LoRA Adapter Parameters

These control **what** gets trained and how much:

| Parameter | Value | What it does |
|-----------|-------|--------------|
| `r` | 16 | LoRA rank — size of the tiny trainable matrices injected into the model |
| `lora_alpha` | 16 | Scaling factor (alpha/r = 1.0) — how strongly LoRA weights influence output |
| `lora_dropout` | 0 | No dropout — optimized for speed |
| `bias` | `"none"` | Don't train bias terms — saves memory |
| `target_modules` | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | The 7 weight matrices inside each Transformer layer that LoRA actually touches |
| `gradient_checkpointing` | `"unsloth"` | Recomputes activations during backprop instead of storing them — saves ~30% VRAM |

### Training Hyperparameters

These control **how** the training loop runs:

| Parameter | Value | Why |
|-----------|-------|-----|
| `per_device_train_batch_size` | 2 | Small batch to fit in 6GB VRAM |
| `gradient_accumulation_steps` | 4 | Simulates effective batch size of 2×4 = 8 |
| `max_steps` | 60 | Short fine-tune run |
| `warmup_steps` | 5 | LR ramps up over first 5 steps — avoids exploding gradients |
| `learning_rate` | 2e-4 (0.0002) | Standard rate for LoRA fine-tuning |
| `lr_scheduler_type` | `"linear"` | LR linearly decays from 2e-4 → 0 by the end |
| `optim` | `"adamw_8bit"` | 8-bit AdamW optimizer — uses 75% less memory than standard Adam |
| `weight_decay` | 0.01 | L2 regularization to prevent overfitting |
| `fp16 / bf16` | auto (bf16 on RTX 4050) | Brain Float 16 precision — faster + less VRAM than fp32 |
| `seed` | 3407 | Fixed random seed for reproducibility |
| `dataset_num_proc` | 2 | 2 CPU threads for dataset preprocessing |
| `max_seq_length` | 2048 | Max token length per training sample |

### Model Loading Parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| `load_in_4bit` | `True` | NF4 quantization — shrinks 7.5GB model to ~2.5GB in VRAM |
| `dtype` | `None` | Auto-detects bfloat16 for RTX 4050 |
| `model_name` | `unsloth/Phi-3-mini-4k-instruct` | The 3.8B base model |

### Dataset

| Setting | Value |
|---------|-------|
| Dataset | `yahma/alpaca-cleaned` |
| Size | 52,000 instruction-following examples |
| Format | `### Instruction:\n{}\n### Response:\n{}` |
| Split | `train` (full set) |

---

## How the Model Was Trained — Step by Step

### Step 1 — Download the Base Model

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Phi-3-mini-4k-instruct",
    max_seq_length = 2048,
    dtype = None,         # auto → bfloat16 on RTX 4050
    load_in_4bit = True,  # compresses 7.5GB → ~2.5GB in VRAM
)
```

Unsloth downloads Microsoft's Phi-3 Mini from HuggingFace and loads it into your GPU using NF4 4-bit quantization. Instead of each weight being stored as a 16-bit float (2 bytes), it gets compressed to 4-bit (0.5 bytes) — 4× smaller with minimal quality loss.

---

### Step 2 — Inject LoRA Adapters

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
)
```

Instead of touching the 3.8B frozen original weights, Unsloth inserts tiny new matrices (A and B) beside each of the 7 projection layers inside all 32 Transformer blocks.

```
Original (FROZEN):     W  →  output
With LoRA (TRAINABLE): W + (B × A) → output
                            ↑
                     Only these ~50M params get trained
```

The original `W` is never changed. Only `A` and `B` (rank 16, tiny matrices) learn from the data.

---

### Step 3 — Prepare the Dataset

```python
from datasets import load_dataset

alpaca_prompt = """Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs      = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = alpaca_prompt.format(instruction, output) + EOS_TOKEN
        texts.append(text)
    return { "text": texts }

dataset = load_dataset("yahma/alpaca-cleaned", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)
```

52,000 instruction-response pairs from the Alpaca dataset were loaded and formatted into a consistent template. The `EOS_TOKEN` at the end teaches the model when to stop generating. Each training example looked like:

```
Below is an instruction that describes a task.

### Instruction:
Explain Newton's second law of motion.

### Response:
Newton's second law states that force equals mass times acceleration (F = ma)....<EOS>
```

---

### Step 4 — The Training Loop

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        bf16 = True,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer.train()
```

What happened internally during each of the 60 steps:

```
①  Pick a batch of 2 training examples
        │
②  Forward pass — run the text through the model
   to get predicted next tokens
        │
③  Calculate Loss — how wrong were the predictions?
   (Cross-entropy loss between predicted vs actual tokens)
        │
④  Backward pass — compute gradients
   (only flows through the LoRA A & B matrices, not the frozen W)
        │
⑤  Gradient accumulation — do this 4 times,
   then update weights (simulates batch size 8)
        │
⑥  AdamW optimizer step — nudge A & B weights
   slightly in the direction that reduces loss
        │
⑦  LR scheduler — slowly reduce learning rate
   as training progresses
        │
   Repeat for 60 steps total
```

---

### Step 5 — Export as GGUF for LM Studio

```python
model.save_pretrained_gguf(
    "my_custom_model",
    tokenizer,
    quantization_method = "q4_k_m"
)
```

The trained LoRA adapters were **merged back into the base model weights** (`W_new = W + B×A`), then the full merged model was re-quantized into compact GGUF format using Q4_K_M — a smart mixed-precision scheme where important layers keep slightly more precision.

| Format | Size | Notes |
|--------|------|-------|
| Original bfloat16 safetensors | ~7.6 GB | Full precision, 2 shard files |
| Q4_K_M GGUF | ~2.15 GB | 72% smaller, runs on CPU or any GPU |

---

## Full Training Timeline

```
Download Phi-3 base (3.8B)      ~5-10 min  (one time only)
        ↓
Inject LoRA adapters            <1 second
        ↓
Load + format Alpaca dataset    ~2-3 min
        ↓
60 training steps (RTX 4050)   ~5-15 min
        ↓
Merge LoRA into full model      ~2 min
        ↓
Export to Q4_K_M GGUF           ~10-20 min
        ↓
Load in LM Studio → App uses it ✅

Total: ~30-40 minutes on RTX 4050
```

---

## How Both Models Work Together in the App

```python
# Step 1: Whisper converts audio → raw text
transcript = transcribe_audio(audio_path, stt_model)

# Step 2: Text is stuffed into a prompt
prompt = f"Convert this transcript into structured notes:\n{transcript}"

# Step 3: Phi-3 (running in LM Studio) generates the notes
stream = client.chat.completions.create(
    model = "local-model",   # your .gguf loaded in LM Studio
    messages = [{"role": "user", "content": prompt}],
    stream = True
)
```

LM Studio acts as a local server that serves your `.gguf` model through an OpenAI-compatible API at `http://localhost:1234/v1` — the app talks to it exactly like it would talk to ChatGPT, just fully offline.

---

## Build Pipeline Summary

```
train_model.py   →  Phi-3 Base + LoRA training  →  my_custom_model/ (safetensors)
export_gguf.py   →  Q4_K_M quantization          →  my_custom_model.Q4_K_M.gguf
LM Studio        →  loads the .gguf              →  serves at localhost:1234
app.py           →  Whisper (audio → text)
                 →  Phi-3 via LM Studio (text → notes)
                 →  Chat powered by same Phi-3
```

Key fact: Only **~1.6% of the 3.8B weights** were actually trained — the LoRA adapters on 7 projection layers × 32 Transformer blocks. The rest (98.4%) stayed frozen. This is what makes it possible to fine-tune on a 6GB laptop GPU in minutes.

---

## Key Libraries Used

| Tool | Version | Role |
|------|---------|------|
| `unsloth` | 2026.4.4 | Fast QLoRA fine-tuning (2–5x faster than baseline HuggingFace) |
| `transformers` | 4.57.3 | Model loading, tokenization, inference pipelines |
| `trl` | latest | `SFTTrainer` for supervised fine-tuning |
| `torch` | 2.9.1 | Core deep learning backend |
| `torchaudio` | 2.11.0 | Audio processing utilities |
| `soundfile` | 0.13.1 | Audio file reading |
| `imageio-ffmpeg` | 0.6.0 | Bundled ffmpeg binary for audio decoding |
| `accelerate` | 1.13.0 | Mixed precision + distributed training |
| `faster-whisper` | latest | Optimized Whisper STT inference |
| `streamlit` | 1.53.0 | Web application UI |

---

## References

- [Phi-3 Technical Report — Microsoft](https://arxiv.org/abs/2404.14219)
- [LoRA: Low-Rank Adaptation of LLMs — Hu et al.](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning — Dettmers et al.](https://arxiv.org/abs/2305.14314)
- [Whisper: Robust Speech Recognition — OpenAI](https://arxiv.org/abs/2212.04356)
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [Alpaca Dataset](https://huggingface.co/datasets/yahma/alpaca-cleaned)
