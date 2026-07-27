# 7B Custom Model — Training & Architecture Details

This document covers all the configuration, hardware strategies, and hyperparameter details used to fine-tune the 7-Billion parameter language model.

---

## 1. Model Overview

- **Base Model:** `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- **Model Size:** 7 Billion parameters
- **Format:** Fine-tuned using LoRA (Low-Rank Adaptation) and exported for local inference.
- **Output Artifacts:**
  - Merged Safetensors: `my_7b_model/`
  - GGUF File (Q4_K_M): `my_7b_model_gguf/`
- **Dataset:** `yahma/alpaca-cleaned`

---

## 2. Hardware & Memory Strategy

Fine-tuning a 7B model locally on an NVIDIA RTX 4050 (6 GB VRAM) requires memory optimization. The script utilizes a **Hybrid GPU + CPU RAM Strategy**:

- **VRAM Budget:** `5500MiB` (Leaves ~500 MB headroom to prevent Out-Of-Memory crashes).
- **RAM Budget:** `16GiB` (System RAM is used for layers that overflow the GPU).
- **4-bit NF4 Quantization:** Shrinks the base model size from ~14 GB down to ~4.1 GB.
- **Device Mapping:** By setting `max_memory` limits, the system automatically splits and offloads model layers across GPU VRAM and CPU RAM.
- **Gradient Checkpointing:** Recomputes intermediate activations during the backward pass instead of storing them, trading some compute time for massive memory savings.

---

## 3. LoRA (Low-Rank Adaptation) Settings

Instead of training all 7 billion parameters, we train tiny adapter matrices injected into the attention and feed-forward layers.

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Rank (`r`)** | `32` | Size of the trainable matrices. Increased from 16 (used in smaller models) for higher capacity in the 7B model. |
| **Alpha** | `32` | Scaling factor for the LoRA weights. |
| **Target Modules** | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | The 7 projection layers modified in each Transformer block. |
| **Dropout** | `0` | Set to 0 for maximum training speed. |
| **Bias** | `"none"` | No biases are trained to save memory. |

---

## 4. Training Hyperparameters

The training was driven by the `SFTTrainer` with the following key settings:

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Max Sequence Length** | `1024` | Maximum token length per training sample. |
| **Max Steps** | `200` | Number of training iterations. |
| **Batch Size** | `1` | Per device batch size (kept small due to 7B size). |
| **Gradient Accumulation** | `4` | Simulates an effective batch size of 4 (1 × 4). |
| **Learning Rate** | `2e-4` (0.0002) | Standard rate for LoRA fine-tuning. |
| **Warmup Steps** | `10` | Gradually ramps up the learning rate to avoid exploding gradients. |
| **LR Scheduler** | `"cosine"` | Decays the learning rate using a cosine curve, which yields better performance for 7B models than linear decay. |
| **Optimizer** | `"adamw_8bit"` | Uses 75% less memory than standard Adam. |
| **Weight Decay** | `0.01` | L2 regularization to prevent overfitting. |
| **Max Grad Norm** | `0.3` | Gradient clipping threshold; crucial for stabilizing 7B training. |
| **Precision** | `bf16` / `fp16` | Automatically uses Brain Float 16 if supported by the GPU. |
| **Seed** | `3407` | Fixed seed for reproducibility. |

---

## 5. Build Pipeline Summary

1. **Initialization:** Hardware limits are detected and enforced.
2. **Model Loading:** The `mistral-7b-instruct` base model is loaded in 4-bit, distributing layers between GPU and CPU based on the predefined memory budgets.
3. **Adapter Injection:** LoRA adapters (rank 32) are attached to the model's projection layers.
4. **Dataset Preparation:** The 52,000-sample Alpaca dataset is loaded and formatted into the instruction-response structure.
5. **Training Loop:** The model undergoes 200 optimization steps utilizing cosine learning rate scheduling and 8-bit AdamW.
6. **Merging & Export:**
   - The LoRA adapters are merged back into the base 16-bit weights.
   - The final model is quantized down to **Q4_K_M** (GGUF format), achieving an optimal balance between file size and model intelligence.
7. **Deployment:** The resulting `.gguf` file can be directly loaded into LM Studio for local, offline inference.
