# NotebookCore-120M — Complete Pipeline Commands

## Training Pipeline (run in order)

```bash
# Step 1: Download datasets (lite mode ~2 GB, ~15-30 min)
python -X utf8 -m model.data.download_datasets

# Step 2: Clean data (~2-5 min)
python -X utf8 -m model.data.clean_data

# Step 3: Train tokenizer (~5-10 min)
python -X utf8 -m model.tokenizer.train_tokenizer --corpus model/data/cleaned/pretrain_corpus.txt

# Step 4: Prepare pretrain data (~5 min)
python -X utf8 -m model.data.prepare_pretrain

# Step 5: Pretrain (~20-40 min for 1K steps)
python -X utf8 -m model.training.pretrain --max-steps 1000

# Step 6: Prepare instruction data (~2-5 min)
python -X utf8 -m model.data.prepare_instruct

# Step 7: Instruction tune (~10-20 min for 500 steps)
python -X utf8 -m model.training.instruct_tune --max-steps 500

# Step 8: Evaluate
python -X utf8 -m model.evaluation.evaluate --checkpoint checkpoints/instruct/best.pt --perplexity-only

# Step 9: Export to HuggingFace format
python -X utf8 -m model.export.export_hf --checkpoint checkpoints/instruct/best.pt

# Step 10: Convert to GGUF (needs llama.cpp installed)
python -X utf8 -m model.export.convert_gguf --input model/export/notebookcore-120m-hf
```

# Re-train with more steps (will take ~8-12 hours on RTX 4050)
python -X utf8 -m model.training.pretrain --max-steps 50000

# Then re-do instruction tuning
python -X utf8 -m model.training.instruct_tune --max-steps 3000

# Re-train with more steps (will take ~8-12 hours on RTX 4050)
python -X utf8 -m model.training.pretrain --max-steps 50000

# Then re-do instruction tuning
python -X utf8 -m model.training.instruct_tune --max-steps 3000

# Re-export
python -X utf8 -m model.export.export_hf --checkpoint checkpoints/instruct/best.pt
python d:\mega_project-v4\llama.cpp\convert_hf_to_gguf.py d:\mega_project-v4\model\export\notebookcore-120m-hf --outfile d:\mega_project-v4\model\export\gguf\notebookcore-120m-f16.gguf --outtype f16

## OR: Single command to run everything

```bash
python -X utf8 run_pipeline.py
```

## llama.cpp Setup (needed for Step 10)

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
pip install -r requirements.txt
```

## LM Studio Deployment (after Step 10)

```bash
# Copy GGUF file to LM Studio models folder
copy model\export\gguf\notebookcore-120m-Q4_K_M.gguf C:\Users\Anish\.cache\lm-studio\models\notebookcore-120m\

# Open LM Studio → Local Server → Load notebookcore-120m-Q4_K_M → Start Server (port 1234)
```

## Start StudySnap AI (3 terminals)

```bash
# Terminal 1: LM Studio already running on port 1234

# Terminal 2: Backend
cd backend
.\venv\Scripts\Activate.ps1
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
uvicorn main:app --reload

# Terminal 3: Frontend
cd frontend
npm run dev
```

## Verify

```bash
# Check LM Studio is serving
curl http://localhost:1234/v1/models

# Check backend is running
curl http://localhost:8000/api/sources

# Open frontend
# http://localhost:5173
```

## Quick Reference

| Command | What it does | Time |
|---|---|---|
| `python -X utf8 run_pipeline.py` | Quick test (1K pretrain + 500 instruct) | ~1-2 hours |
| `python -X utf8 run_pipeline.py --pretrain-steps 5000 --instruct-steps 2000` | Medium run | ~4-6 hours |
| `python -X utf8 run_pipeline.py --full` | Full training (100K + 5K) | ~24-48 hours |
| `python -X utf8 run_pipeline.py --start-from 5` | Resume from pretrain step | Varies |
| `python -X utf8 run_pipeline.py --steps 1,2,3` | Run specific steps only | ~30 min 