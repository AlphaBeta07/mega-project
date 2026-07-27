"""
multi_model_inference.py
========================
Run multiple AI models simultaneously using GPU VRAM + System RAM.

This script loads two models at the same time and routes tasks intelligently:

  ┌──────────────────────────────────────────────────────────────────┐
  │  FAST MODEL  (GPU VRAM only)                                     │
  │  Gemma 3 4B  →  Quick note generation, short Q&A                │
  ├──────────────────────────────────────────────────────────────────┤
  │  SMART MODEL  (GPU VRAM + RAM overflow)                          │
  │  Mistral 7B or Llama 3.1 8B  →  Deep explanation, complex chat  │
  └──────────────────────────────────────────────────────────────────┘

Memory layout with 6GB VRAM + 16GB RAM:
  Gemma 3 4B  Q4  → ~2.5 GB VRAM
  Mistral 7B  Q4  → ~4.1 GB VRAM
  ─────────────────────────────────
  Combined         ~6.6 GB total
  Strategy: Gemma in VRAM (2.5 GB) + Mistral split ~3.5 GB VRAM + ~0.6 GB RAM

Usage:
  python multi_model_inference.py

Then import in your app:
  from multi_model_inference import router
  reply = router.generate(prompt, mode="notes")  # or mode="chat"
"""

import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from threading import Thread

# =============================================================================
# CONFIGURATION
# =============================================================================

# Fast model — small & quick (Gemma 3 4B or your custom fine-tuned Phi-3)
FAST_MODEL_ID  = "google/gemma-3-4b-it"         # change to "my_custom_model" to use yours

# Smart model — larger & more capable (Mistral 7B, Llama 3.1 8B, Qwen2.5 7B)
SMART_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Memory budgets
GPU_MEMORY_MB  = 5500   # MB of VRAM available
RAM_MEMORY_GB  = 14     # GB of CPU RAM for overflow

# =============================================================================
# SHARED 4-BIT CONFIG
# =============================================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.bfloat16,
    bnb_4bit_use_double_quant = True,   # nested quantization → extra 0.4 GB saved
)

# =============================================================================
# MODEL LOADER UTILITY
# =============================================================================
def load_model(model_id: str, gpu_mb: int, ram_gb: int, label: str):
    """
    Load a model with automatic GPU+RAM splitting.
    gpu_mb  : how many MB of VRAM this model can use
    ram_gb  : CPU RAM overflow budget
    """
    print(f"\n[{label}] Loading {model_id} ...")
    print(f"         VRAM budget : {gpu_mb} MB")
    print(f"         RAM  budget : {ram_gb} GB")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config = bnb_config,
        device_map          = "auto",           # splits across GPU + CPU automatically
        max_memory          = {
            0:     f"{gpu_mb}MiB",              # GPU VRAM limit
            "cpu": f"{ram_gb}GiB",              # CPU RAM limit
        },
        torch_dtype         = torch.bfloat16,
        trust_remote_code   = True,
    )
    model.eval()  # inference mode — disable dropout

    # Report device split
    if hasattr(model, 'hf_device_map'):
        gpu_n = sum(1 for v in model.hf_device_map.values() if v != "cpu")
        cpu_n = sum(1 for v in model.hf_device_map.values() if v == "cpu")
        print(f"[{label}] ✅ Loaded | GPU layers: {gpu_n} | CPU layers: {cpu_n}")
    else:
        print(f"[{label}] ✅ Loaded")

    return model, tokenizer


# =============================================================================
# ROUTER CLASS
# =============================================================================
class ModelRouter:
    """
    Manages two models and routes prompts to the right one.

    Routing logic:
      mode="notes"  → FAST model  (Gemma 4B — quick structured notes)
      mode="chat"   → SMART model (Mistral 7B — deep Q&A)
      mode="auto"   → SMART model if prompt > 300 chars, else FAST
    """

    def __init__(self):
        self.fast_model     = None
        self.fast_tokenizer = None
        self.smart_model    = None
        self.smart_tokenizer= None
        self._loaded        = False

    def load(self):
        if self._loaded:
            return

        print("\n" + "="*60)
        print("  LOADING MULTI-MODEL ENGINE")
        print("="*60)

        # Split VRAM: give Gemma 3B, Mistral gets the rest (3GB = 3072 MB + RAM overflow)
        fast_vram  = 3000
        smart_vram = GPU_MEMORY_MB - fast_vram   # ~2500 MB on GPU, rest to RAM

        self.fast_model,  self.fast_tokenizer  = load_model(
            FAST_MODEL_ID,  fast_vram,  RAM_MEMORY_GB, "FAST  (Gemma 4B)"
        )
        self.smart_model, self.smart_tokenizer = load_model(
            SMART_MODEL_ID, smart_vram, RAM_MEMORY_GB, "SMART (Mistral 7B)"
        )

        self._loaded = True
        print("\n✅ Both models ready. Router online.\n")

    def _build_messages(self, prompt: str, system: str) -> list:
        return [
            {"role": "system",    "content": system},
            {"role": "user",      "content": prompt},
        ]

    def _stream_generate(self, model, tokenizer, messages: list, max_new_tokens=1024):
        """Stream tokens one by one using TextIteratorStreamer."""
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True,
            return_tensors        = "pt",
        ).to(model.device)

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt         = True,
            skip_special_tokens = True,
        )
        gen_kwargs = dict(
            input_ids  = input_ids,
            streamer   = streamer,
            max_new_tokens = max_new_tokens,
            do_sample  = True,
            temperature= 0.7,
            top_p      = 0.9,
            repetition_penalty = 1.1,
        )
        # Run generation in background thread so we can stream output
        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        for token in streamer:
            yield token

        thread.join()

    def generate(self, prompt: str, mode: str = "auto", system: str = None):
        """
        Generate a response.

        Args:
            prompt : The user's text input
            mode   : "notes"  → fast model (Gemma)
                     "chat"   → smart model (Mistral)
                     "auto"   → smart if prompt > 300 chars, else fast
            system : Optional system prompt override

        Yields:
            str tokens one by one (streaming)
        """
        if not self._loaded:
            self.load()

        # Choose model
        if mode == "notes":
            model, tokenizer = self.fast_model, self.fast_tokenizer
            label = "Gemma 4B (Fast)"
        elif mode == "chat":
            model, tokenizer = self.smart_model, self.smart_tokenizer
            label = "Mistral 7B (Smart)"
        else:  # auto
            if len(prompt) > 300:
                model, tokenizer = self.smart_model, self.smart_tokenizer
                label = "Mistral 7B (Smart — auto)"
            else:
                model, tokenizer = self.fast_model, self.fast_tokenizer
                label = "Gemma 4B (Fast — auto)"

        if system is None:
            system = "You are a helpful educational AI assistant. Answer clearly using markdown."

        messages = self._build_messages(prompt, system)
        print(f"[Router] Using: {label}")

        yield from self._stream_generate(model, tokenizer, messages)

    def generate_notes(self, transcript: str):
        """Specialized: generate structured study notes from a transcript."""
        system = (
            "You are an expert educational assistant. "
            "Convert lecture transcripts into clear, structured Markdown study notes."
        )
        prompt = f"""Convert the following lecture transcript into structured educational notes.
Use this exact format:

# Title
## Key Points
## Detailed Explanation
## Examples
## Summary
## Possible Exam Questions

Transcript:
{transcript[:4000]}
"""
        yield from self.generate(prompt, mode="notes", system=system)

    def chat(self, messages_history: list, context: str = None):
        """Specialized: multi-turn chat with optional notes context."""
        system = "You are a helpful educational AI assistant. Use markdown for formatting."
        if context:
            system += f"\n\nUse this lecture context to answer questions:\n{context[:2000]}"

        # Build the full messages list
        full_messages = [{"role": "system", "content": system}]
        full_messages += messages_history[-10:]  # keep last 10 turns

        if not self._loaded:
            self.load()

        input_ids = self.smart_tokenizer.apply_chat_template(
            full_messages,
            add_generation_prompt = True,
            return_tensors        = "pt",
        ).to(self.smart_model.device)

        streamer = TextIteratorStreamer(
            self.smart_tokenizer,
            skip_prompt         = True,
            skip_special_tokens = True,
        )
        thread = Thread(target=self.smart_model.generate, kwargs=dict(
            input_ids      = input_ids,
            streamer       = streamer,
            max_new_tokens = 512,
            do_sample      = True,
            temperature    = 0.7,
            top_p          = 0.9,
        ))
        thread.start()
        for token in streamer:
            yield token
        thread.join()


# =============================================================================
# GLOBAL ROUTER INSTANCE (import this in app.py)
# =============================================================================
router = ModelRouter()


# =============================================================================
# QUICK TEST (run this file directly to test)
# =============================================================================
if __name__ == "__main__":
    print("Loading models for test...")
    router.load()

    print("\n--- TEST 1: Notes generation (Gemma 4B) ---")
    test_transcript = (
        "Today we talked about photosynthesis. Plants convert sunlight into glucose "
        "using chlorophyll. The process happens in the chloroplasts. CO2 and water "
        "are inputs. Oxygen and glucose are outputs. Light reactions happen in the "
        "thylakoids. The Calvin cycle happens in the stroma."
    )
    result = ""
    for token in router.generate_notes(test_transcript):
        print(token, end="", flush=True)
        result += token

    print("\n\n--- TEST 2: Deep chat (Mistral 7B) ---")
    messages = [{"role": "user", "content": "Explain the Calvin cycle in detail with equations."}]
    for token in router.chat(messages):
        print(token, end="", flush=True)

    print("\n\nAll tests passed ✅")
