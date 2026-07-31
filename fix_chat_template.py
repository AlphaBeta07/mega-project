"""Add chat template to tokenizer_config.json for LM Studio compatibility."""
import json
import os

path = os.path.join(os.path.dirname(__file__), "model", "export", "notebookcore-200m-hf", "tokenizer_config.json")

with open(path, "r") as f:
    config = json.load(f)

# Jinja2 chat template matching our training format:
# <|system|>\nYou are...\n<|endofturn|>\n<|user|>\nQuestion\n<|endofturn|>\n<|assistant|>\nAnswer\n<|endofturn|>
chat_template = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|system|>\n{{ message['content'] }}\n<|endofturn|>\n"
    "{% elif message['role'] == 'user' %}"
    "<|user|>\n{{ message['content'] }}\n<|endofturn|>\n"
    "{% elif message['role'] == 'assistant' %}"
    "<|assistant|>\n{{ message['content'] }}\n<|endofturn|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|assistant|>\n"
    "{% endif %}"
)

config["chat_template"] = chat_template
config["add_bos_token"] = False
config["eos_token"] = "<|endofturn|>"  # CRITICAL: Tell LM Studio to stop at <|endofturn|>

with open(path, "w") as f:
    json.dump(config, f, indent=2)

# Create added_tokens.json to map custom tokens to their trained IDs (3-7)
added_tokens_path = os.path.join(os.path.dirname(__file__), "model", "export", "notebookcore-200m-hf", "added_tokens.json")
added_tokens = {
    "<|pad|>": 3,
    "<|user|>": 4,
    "<|assistant|>": 5,
    "<|system|>": 6,
    "<|endofturn|>": 7
}
with open(added_tokens_path, "w") as f:
    json.dump(added_tokens, f, indent=2)

print("Chat template added to tokenizer_config.json")
print("added_tokens.json generated with custom token mappings")
print(f"Template: {chat_template[:100]}...")
