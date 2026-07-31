"""
run_pipeline.py -- Master Pipeline for NotebookCore-120M
==========================================================
Runs the entire training pipeline end-to-end:

    Step 1: Download datasets (lite mode, ~2 GB)
    Step 2: Clean data
    Step 3: Train tokenizer (32K BPE)
    Step 4: Prepare pretraining data (tokenize + shard)
    Step 5: Pretrain (next-token prediction)
    Step 6: Prepare instruction data
    Step 7: Instruction tune (SFT)
    Step 8: Evaluate
    Step 9: Export to HuggingFace
    Step 10: Convert to GGUF

Disk Budget: ~5 GB (fits within 10 GB)
Hardware: RTX 4050 (6 GB VRAM), 16 GB RAM

Usage:
    python run_pipeline.py                    # Full pipeline
    python run_pipeline.py --start-from 5     # Resume from step 5 (pretrain)
    python run_pipeline.py --steps 1,2,3      # Run only specific steps
    python run_pipeline.py --pretrain-steps 500  # Quick test with 500 pretrain steps
"""

import os
import sys
import time
import subprocess
import argparse

# Fix Windows encoding for Unicode characters in print statements
os.environ["PYTHONIOENCODING"] = "utf-8"

# Ensure we run from the project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)


def run_step(step_num, name, command, cwd=None):
    """Run a pipeline step and report status."""
    print(f"\n{'='*60}")
    print(f"  STEP {step_num}: {name}")
    print(f"{'='*60}")
    print(f"  Command: {command}")
    print(f"  Started: {time.strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")

    start = time.time()

    result = subprocess.run(
        command,
        shell=True,
        cwd=cwd or PROJECT_ROOT,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )

    elapsed = time.time() - start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    if result.returncode == 0:
        print(f"\n  [OK] Step {step_num} completed in {minutes}m {seconds}s")
    else:
        print(f"\n  [FAIL] Step {step_num} failed (exit code {result.returncode})")
        print(f"  You can retry this step with: python run_pipeline.py --start-from {step_num}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="NotebookCore-120M -- Full Training Pipeline")
    parser.add_argument("--start-from", type=int, default=1,
                        help="Start from step N (skip earlier steps)")
    parser.add_argument("--steps", type=str, default=None,
                        help="Run only these steps (comma-separated, e.g. '1,2,3')")
    parser.add_argument("--pretrain-steps", type=int, default=1000,
                        help="Number of pretraining steps (default: 1000, full: 100000)")
    parser.add_argument("--instruct-steps", type=int, default=500,
                        help="Number of instruction tuning steps (default: 500, full: 5000)")
    parser.add_argument("--full", action="store_true",
                        help="Full training (100K pretrain + 5K instruct steps)")
    args = parser.parse_args()

    if args.full:
        args.pretrain_steps = 100_000
        args.instruct_steps = 5_000

    # Determine which steps to run
    if args.steps:
        steps_to_run = set(int(s) for s in args.steps.split(","))
    else:
        steps_to_run = set(range(args.start_from, 11))

    print("=" * 60)
    print("  NotebookCore-120M -- MASTER PIPELINE")
    print("=" * 60)
    print(f"  Pretrain steps  : {args.pretrain_steps:,}")
    print(f"  Instruct steps  : {args.instruct_steps:,}")
    print(f"  Steps to run    : {sorted(steps_to_run)}")
    print(f"  Working dir     : {PROJECT_ROOT}")
    print("=" * 60)

    pipeline_start = time.time()

    # ── STEP 1: Download datasets ──
    if 1 in steps_to_run:
        ok = run_step(1, "DOWNLOAD DATASETS (lite mode)",
                      "python -m model.data.download_datasets")
        if not ok:
            return

    # ── STEP 2: Clean data ──
    if 2 in steps_to_run:
        ok = run_step(2, "CLEAN DATA",
                      "python -m model.data.clean_data")
        if not ok:
            return

    # ── STEP 3: Train tokenizer ──
    if 3 in steps_to_run:
        corpus_path = os.path.join("model", "data", "cleaned", "pretrain_corpus.txt")
        ok = run_step(3, "TRAIN TOKENIZER (32K BPE)",
                      f"python -m model.tokenizer.train_tokenizer --corpus {corpus_path}")
        if not ok:
            return

    # ── STEP 4: Prepare pretraining data ──
    if 4 in steps_to_run:
        ok = run_step(4, "PREPARE PRETRAINING DATA",
                      "python -m model.data.prepare_pretrain")
        if not ok:
            return

    # ── STEP 5: Pretrain ──
    if 5 in steps_to_run:
        ok = run_step(5, f"PRETRAIN ({args.pretrain_steps:,} steps)",
                      f"python -m model.training.pretrain --max-steps {args.pretrain_steps}")
        if not ok:
            return

    # ── STEP 6: Prepare instruction data ──
    if 6 in steps_to_run:
        ok = run_step(6, "PREPARE INSTRUCTION DATA",
                      "python -m model.data.prepare_instruct")
        if not ok:
            return

    # ── STEP 7: Instruction tune ──
    if 7 in steps_to_run:
        ok = run_step(7, f"INSTRUCTION TUNE ({args.instruct_steps:,} steps)",
                      f"python -m model.training.instruct_tune --max-steps {args.instruct_steps}")
        if not ok:
            return

    # ── STEP 8: Evaluate ──
    if 8 in steps_to_run:
        checkpoint = os.path.join("checkpoints", "instruct", "best.pt")
        if not os.path.exists(checkpoint):
            checkpoint = os.path.join("checkpoints", "pretrain", "best.pt")
        if os.path.exists(checkpoint):
            ok = run_step(8, "EVALUATE",
                          f"python -m model.evaluation.evaluate --checkpoint {checkpoint} --perplexity-only")
        else:
            print("\n  [SKIP] Step 8: No checkpoint found to evaluate")

    # ── STEP 9: Export to HuggingFace ──
    if 9 in steps_to_run:
        checkpoint = os.path.join("checkpoints", "instruct", "best.pt")
        if not os.path.exists(checkpoint):
            checkpoint = os.path.join("checkpoints", "pretrain", "best.pt")
        if os.path.exists(checkpoint):
            ok = run_step(9, "EXPORT TO HUGGINGFACE",
                          f"python -m model.export.export_hf --checkpoint {checkpoint}")
        else:
            print("\n  [SKIP] Step 9: No checkpoint found to export")

    # ── STEP 10: Convert to GGUF ──
    if 10 in steps_to_run:
        hf_dir = os.path.join("model", "export", "notebookcore-120m-hf")
        if os.path.exists(hf_dir):
            ok = run_step(10, "CONVERT TO GGUF",
                          f"python -m model.export.convert_gguf --input {hf_dir}")
        else:
            print("\n  [SKIP] Step 10: No HuggingFace export found")

    # ── SUMMARY ──
    total_time = time.time() - pipeline_start
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Total time: {hours}h {minutes}m {seconds}s")
    print(f"  Steps run : {sorted(steps_to_run)}")
    print(f"{'='*60}")
    print()
    print("  Next steps:")
    print("    1. Copy the GGUF file to LM Studio models directory")
    print("    2. Load in LM Studio and start the server")
    print("    3. Start the backend:  cd backend && python -m uvicorn main:app --port 8000")
    print("    4. Start the frontend: cd frontend && npm run dev")
    print()


if __name__ == "__main__":
    main()
