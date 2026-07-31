# NotebookCore-120M — Build Walkthrough

## Summary

Built the complete **NotebookCore-120M** training pipeline from scratch — 25 Python source files across 8 components, totaling ~120KB of production-grade code.

## Files Created (25 total)

### Configuration (1 file)
| File | Size | Purpose |
|---|---|---|
| [config.py](file:///d:/mega_project-v4/model/config.py) | 10 KB | `NotebookCoreConfig`, `PretrainConfig`, `InstructConfig`, `SpecialTokens` |

### Tokenizer (2 files)
| File | Size | Purpose |
|---|---|---|
| [train_tokenizer.py](file:///d:/mega_project-v4/model/tokenizer/train_tokenizer.py) | 9 KB | SentencePiece BPE training (32K vocab, custom special tokens) |
| [\_\_init\_\_.py](file:///d:/mega_project-v4/model/tokenizer/__init__.py) | — | Package init |

### Architecture (7 files)
| File | Size | Purpose |
|---|---|---|
| [rope.py](file:///d:/mega_project-v4/model/architecture/rope.py) | 6 KB | Rotary Position Embeddings |
| [rmsnorm.py](file:///d:/mega_project-v4/model/architecture/rmsnorm.py) | 3 KB | Root Mean Square Normalization |
| [attention.py](file:///d:/mega_project-v4/model/architecture/attention.py) | 6 KB | Causal Multi-Head Self-Attention + RoPE |
| [swiglu.py](file:///d:/mega_project-v4/model/architecture/swiglu.py) | 5 KB | SwiGLU gated FFN |
| [transformer_block.py](file:///d:/mega_project-v4/model/architecture/transformer_block.py) | 6 KB | Pre-norm decoder block |
| [model.py](file:///d:/mega_project-v4/model/architecture/model.py) | 15 KB | Full model (embed → 12×block → norm → LM head) |
| [\_\_init\_\_.py](file:///d:/mega_project-v4/model/architecture/__init__.py) | — | Package init with exports |

### Data Pipeline (5 files)
| File | Size | Purpose |
|---|---|---|
| [download_datasets.py](file:///d:/mega_project-v4/model/data/download_datasets.py) | 8 KB | HuggingFace dataset download (streaming) |
| [clean_data.py](file:///d:/mega_project-v4/model/data/clean_data.py) | 12 KB | 5-stage cleaning pipeline |
| [prepare_pretrain.py](file:///d:/mega_project-v4/model/data/prepare_pretrain.py) | 7 KB | Tokenize + shard for pretraining |
| [prepare_instruct.py](file:///d:/mega_project-v4/model/data/prepare_instruct.py) | 13 KB | Chat template formatting + masked labels |
| [\_\_init\_\_.py](file:///d:/mega_project-v4/model/data/__init__.py) | — | Package init |

### Training (4 files)
| File | Size | Purpose |
|---|---|---|
| [utils.py](file:///d:/mega_project-v4/model/training/utils.py) | 8 KB | Checkpointing, LR schedule, GPU monitoring, logger |
| [pretrain.py](file:///d:/mega_project-v4/model/training/pretrain.py) | 11 KB | Pretraining loop (FP16, grad checkpoint, cosine LR) |
| [instruct_tune.py](file:///d:/mega_project-v4/model/training/instruct_tune.py) | 10 KB | SFT loop with masked loss |
| [\_\_init\_\_.py](file:///d:/mega_project-v4/model/training/__init__.py) | — | Package init |

### Evaluation (2 files)
| File | Size | Purpose |
|---|---|---|
| [evaluate.py](file:///d:/mega_project-v4/model/evaluation/evaluate.py) | 10 KB | Perplexity, generation samples, instruction eval |
| [\_\_init\_\_.py](file:///d:/mega_project-v4/model/evaluation/__init__.py) | — | Package init |

### Export (3 files)
| File | Size | Purpose |
|---|---|---|
| [export_hf.py](file:///d:/mega_project-v4/model/export/export_hf.py) | 9 KB | HuggingFace safetensors + config export |
| [convert_gguf.py](file:///d:/mega_project-v4/model/export/convert_gguf.py) | 9 KB | llama.cpp GGUF conversion + Q4_K_M quantization |
| [\_\_init\_\_.py](file:///d:/mega_project-v4/model/export/__init__.py) | — | Package init |

### Package Root (1 file)
| File | Purpose |
|---|---|
| [\_\_init\_\_.py](file:///d:/mega_project-v4/model/__init__.py) | Root package init |

## Verification Results

| Test | Result |
|---|---|
| Config imports | ✅ All 4 config classes import correctly |
| Model instantiation | ✅ 137,841,408 raw parameters |
| Weight tying | ✅ Embedding and LM head share same tensor object |
| Effective parameters | ✅ 113,265,408 (~113.3M unique) |
| Forward pass | ✅ logits shape `(2, 128, 32000)`, loss = 10.50 |
| Generation | ✅ Autoregressive generation produces valid token IDs |

## End-to-End Pipeline

```
Step 1: Download datasets
    python -m model.data.download_datasets

Step 2: Clean data
    python -m model.data.clean_data

Step 3: Train tokenizer
    python -m model.tokenizer.train_tokenizer --corpus model/data/cleaned/pretrain_corpus.txt

Step 4: Prepare pretraining data
    python -m model.data.prepare_pretrain

Step 5: Pretrain
    python -m model.training.pretrain

Step 6: Prepare instruction data
    python -m model.data.prepare_instruct

Step 7: Instruction tune
    python -m model.training.instruct_tune

Step 8: Evaluate
    python -m model.evaluation.evaluate --checkpoint checkpoints/instruct/best.pt

Step 9: Export to HuggingFace
    python -m model.export.export_hf --checkpoint checkpoints/instruct/best.pt

Step 10: Convert to GGUF
    python -m model.export.convert_gguf --input model/export/notebookcore-120m-hf

Step 11: Load in LM Studio → Start server → StudySnap AI connects
```
