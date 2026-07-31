"""
batch_train.py — Batch Training for NotebookCore-200M
=====================================================
Splits 100,000 pretrain steps into batches of ~10,000 steps each.
Each batch resumes from the previous checkpoint automatically.

Benefits:
    - VRAM is fully freed between batches (no fragmentation)
    - Progress is saved after every batch
    - If a batch crashes, you only lose that batch
    - Can stop and resume at any batch boundary

Usage:
    python -X utf8 batch_train.py                         # Full 100K pretrain
    python -X utf8 batch_train.py --start-batch 5         # Resume from batch 5
    python -X utf8 batch_train.py --batch-size 8000       # 8K steps per batch
    python -X utf8 batch_train.py --skip-pretrain         # Only do instruct tuning
"""

import os
import sys
import time
import subprocess
import argparse
import glob

os.environ["PYTHONIOENCODING"] = "utf-8"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in a directory."""
    best = os.path.join(checkpoint_dir, "best.pt")
    if os.path.exists(best):
        return best

    # Find step_*.pt files and pick the highest
    step_files = glob.glob(os.path.join(checkpoint_dir, "step_*.pt"))
    if step_files:
        step_files.sort(key=lambda f: int(f.split("step_")[1].split(".pt")[0]))
        return step_files[-1]

    return None


def run_batch(batch_num, total_batches, max_steps, checkpoint_dir, phase="pretrain"):
    """Run a single training batch."""
    print(f"\n{'='*60}")
    print(f"  BATCH {batch_num}/{total_batches} — {phase.upper()}")
    print(f"  Target: step {max_steps:,}")
    print(f"{'='*60}")

    # Find checkpoint to resume from
    checkpoint = find_latest_checkpoint(checkpoint_dir)
    resume_arg = f"--resume {checkpoint}" if checkpoint else ""

    if phase == "pretrain":
        cmd = f"python -X utf8 -m model.training.pretrain --max-steps {max_steps} {resume_arg}"
    else:
        # instruct_tune.py uses --checkpoint, not --resume
        checkpoint_arg = f"--checkpoint {checkpoint}" if checkpoint else ""
        cmd = f"python -X utf8 -m model.training.instruct_tune --max-steps {max_steps} {checkpoint_arg}"

    print(f"  Command: {cmd}")
    print(f"  Resume:  {checkpoint or 'Fresh start'}")
    print(f"  Started: {time.strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    elapsed = time.time() - start
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)

    if result.returncode == 0:
        print(f"\n  [OK] Batch {batch_num} completed in {mins}m {secs}s")
        return True
    else:
        print(f"\n  [FAIL] Batch {batch_num} failed (exit code {result.returncode})")
        print(f"  Retry with: python -X utf8 batch_train.py --start-batch {batch_num}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch training for NotebookCore-200M")
    parser.add_argument("--total-steps", type=int, default=100_000,
                        help="Total pretraining steps (default: 100000)")
    parser.add_argument("--batch-size", type=int, default=10_000,
                        help="Steps per batch (default: 10000)")
    parser.add_argument("--start-batch", type=int, default=1,
                        help="Start from this batch number (for resuming)")
    parser.add_argument("--instruct-steps", type=int, default=5000,
                        help="Instruction tuning steps (default: 5000)")
    parser.add_argument("--instruct-batches", type=int, default=5,
                        help="Number of instruct batches (default: 5)")
    parser.add_argument("--skip-pretrain", action="store_true",
                        help="Skip pretraining, only do instruction tuning")
    parser.add_argument("--skip-instruct", action="store_true",
                        help="Skip instruction tuning")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip export after training")
    args = parser.parse_args()

    # Calculate batches
    num_pretrain_batches = args.total_steps // args.batch_size
    instruct_batch_size = args.instruct_steps // args.instruct_batches

    print("=" * 60)
    print("  NotebookCore-200M — BATCH TRAINING")
    print("=" * 60)
    print(f"  Pretrain  : {args.total_steps:,} steps in {num_pretrain_batches} batches of {args.batch_size:,}")
    print(f"  Instruct  : {args.instruct_steps:,} steps in {args.instruct_batches} batches of {instruct_batch_size:,}")
    print(f"  Start from: batch {args.start_batch}")
    print("=" * 60)

    pipeline_start = time.time()

    # ── PRETRAINING BATCHES ──
    if not args.skip_pretrain:
        pretrain_dir = os.path.join("checkpoints", "pretrain")
        os.makedirs(pretrain_dir, exist_ok=True)

        for batch in range(args.start_batch, num_pretrain_batches + 1):
            target_step = batch * args.batch_size

            ok = run_batch(
                batch_num=batch,
                total_batches=num_pretrain_batches,
                max_steps=target_step,
                checkpoint_dir=pretrain_dir,
                phase="pretrain",
            )

            if not ok:
                print(f"\n  Pretrain batch {batch} failed. Fix the issue and resume with:")
                print(f"  python -X utf8 batch_train.py --start-batch {batch}")
                return

            # Brief pause to let VRAM fully free
            print("  Clearing GPU memory...")
            time.sleep(5)

    # ── INSTRUCTION TUNING BATCHES ──
    if not args.skip_instruct:
        instruct_dir = os.path.join("checkpoints", "instruct")
        os.makedirs(instruct_dir, exist_ok=True)

        for batch in range(1, args.instruct_batches + 1):
            target_step = batch * instruct_batch_size

            ok = run_batch(
                batch_num=batch,
                total_batches=args.instruct_batches,
                max_steps=target_step,
                checkpoint_dir=instruct_dir,
                phase="instruct",
            )

            if not ok:
                print(f"\n  Instruct batch {batch} failed.")
                return

            time.sleep(5)

    # ── EXPORT ──
    if not args.skip_export:
        print(f"\n{'='*60}")
        print("  EXPORTING MODEL")
        print(f"{'='*60}")

        checkpoint = find_latest_checkpoint(os.path.join("checkpoints", "instruct"))
        if not checkpoint:
            checkpoint = find_latest_checkpoint(os.path.join("checkpoints", "pretrain"))

        if checkpoint:
            # Export to HuggingFace
            subprocess.run(
                f"python -X utf8 -m model.export.export_hf --checkpoint {checkpoint}",
                shell=True, cwd=PROJECT_ROOT,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )

            # Convert to GGUF
            hf_dir = os.path.join("model", "export", "notebookcore-200m-hf")
            gguf_out = os.path.join("model", "export", "gguf", "notebookcore-200m-f16.gguf")
            llama_cpp = os.path.join(PROJECT_ROOT, "llama.cpp", "convert_hf_to_gguf.py")

            if os.path.exists(llama_cpp) and os.path.exists(hf_dir):
                os.makedirs(os.path.dirname(gguf_out), exist_ok=True)
                subprocess.run(
                    f"python {llama_cpp} {hf_dir} --outfile {gguf_out} --outtype f16",
                    shell=True, cwd=PROJECT_ROOT,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                )
                print(f"\n  GGUF saved: {gguf_out}")

    # ── SUMMARY ──
    total = time.time() - pipeline_start
    hours = int(total // 3600)
    mins = int((total % 3600) // 60)

    print(f"\n{'='*60}")
    print(f"  BATCH TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Total time: {hours}h {mins}m")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
