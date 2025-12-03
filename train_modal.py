"""
Modal wrapper for Collatz Lyapunov training.

Usage:
    # First time setup
    pip install modal
    modal setup

    # Run training (pass collatz.py args as a quoted string)
    modal run train_modal.py --args "--epochs 50000 --layers 512,256,128 --curriculum"
    
    # Example matching your local command:
    modal run train_modal.py --args "--epochs 100000 --layers 2048,1024,1024,512,512,512,256,128,64 --curriculum --ramp-len 100000 --start-seeds 0 --mine-negatives"

    # Check results later
    modal volume ls collatz-results
    modal volume get collatz-results /20241202_143052 --out ./results
"""

import modal

image = modal.Image.debian_slim(python_version="3.11").apt_install("curl").pip_install("torch", "numpy")
volume = modal.Volume.from_name("collatz-results", create_if_missing=True)
app = modal.App("collatz-lyapunov", image=image)


@app.function(
    gpu="T4",
    timeout=14400,  # 4 hours max
    volumes={"/results": volume},
)
def train(args: list[str]):
    import subprocess
    from pathlib import Path
    from datetime import datetime

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"/results/{run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path("/root/collatz")
    work_dir.mkdir(exist_ok=True)

    # Download scripts from repo
    repo = "gabrielberger1000/collatz_lyapunov"
    for script in ["collatz.py", "collatz_eval_gen.py"]:
        subprocess.run(["curl", "-sL", f"https://raw.githubusercontent.com/{repo}/main/{script}", 
                        "-o", str(work_dir / script)], check=True)

    # Generate eval set
    print("Generating eval set...")
    subprocess.run(["python", str(work_dir / "collatz_eval_gen.py"), "--samples", "50000",
                    "--output", str(work_dir / "eval.csv"), "--seed", "42"], check=True)

    # Build command - inject eval-csv and save-model, pass everything else through
    cmd = ["python", "-u", str(work_dir / "collatz.py"),
           "--eval-csv", str(work_dir / "eval.csv"),
           "--save-model", str(run_dir / "model.pth")] + args

    # Save config
    (run_dir / "command.txt").write_text(" ".join(["python collatz.py"] + args))

    print(f"Run ID: {run_id}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 70)

    with open(run_dir / "output.log", "w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        proc.wait()

    # Save checkpoints
    import shutil
    for f in list(work_dir.glob("*.pth")) + list(work_dir.glob("*.csv")):
        shutil.copy(f, run_dir)
    
    volume.commit()
    print("=" * 70)
    print(f"Results saved to: /{run_id}/")
    print(f"Retrieve: modal volume get collatz-results /{run_id} --out ./results")


@app.local_entrypoint()
def main(args: str = ""):
    """
    Run with: modal run train_modal.py --args "--epochs 50000 --layers 512,256 --curriculum"
    """
    arg_list = args.split() if args else []
    train.remote(arg_list)
