import modal
import os
import sys

def main():
    vol_name = "collatz-results"
    
    print(f"Connecting to volume '{vol_name}'...")
    try:
        vol = modal.Volume.from_name(vol_name)
    except Exception as e:
        print(f"Error connecting: {e}")
        return

    # 1. List Runs
    print("Fetching run history...")
    try:
        # Get list of run directories
        entries = [e for e in vol.listdir("/", recursive=False) 
                  if e.type == modal.volume.FileEntryType.DIRECTORY]
    except Exception as e:
        print(f"Could not list volume. Run 'modal volume create {vol_name}' if it doesn't exist.")
        return

    if not entries:
        print("No runs found.")
        return

    # Sort by time (newest first)
    entries.sort(key=lambda x: x.mtime, reverse=True)
    latest_run = entries[0]
    run_id = latest_run.path.strip("/")
    
    print(f"\nSTATUS FOR RUN: {run_id}")
    print(f"Started: {latest_run.mtime}")
    print("-" * 40)

    # 2. Check Logs
    log_path = f"{latest_run.path}/output.log"
    model_path = f"{latest_run.path}/model.pth"
    
    try:
        # Read the whole log (it's text, usually small enough)
        # modal.read_file returns an iterator of bytes
        log_bytes = b"".join(vol.read_file(log_path))
        logs = log_bytes.decode("utf-8", errors="replace").splitlines()
        
        if logs:
            # Print the last 5 lines to show progress
            print("LATEST LOG OUTPUT:")
            for line in logs[-5:]:
                print(f"  {line}")
        else:
            print("[Log file exists but is empty]")

        # Check if finished
        files = [e.path for e in vol.listdir(latest_run.path)]
        if any("model.pth" in f for f in files):
            # Check logs for "Saved final model" to be sure
            if any("Saved final model" in l for l in logs[-10:]):
                print("\n✅ STATUS: COMPLETED")
            else:
                print("\n⏳ STATUS: IN PROGRESS (Checkpoint saved)")
        else:
            print("\n⏳ STATUS: IN PROGRESS (No model saved yet)")

    except Exception:
        print("❌ STATUS: UNKNOWN (Could not read output.log)")

    # 3. Download Option
    print("-" * 40)
    q = input(f"Download files for {run_id}? [y/N] > ").strip().lower()
    if q == 'y':
        local_path = os.path.join("results")
        os.makedirs(local_path, exist_ok=True)
        
        print(f"Downloading to {local_path}...")
        # We use os.system because recursive download is easier via CLI
        cmd = f"modal volume get {vol_name} /{run_id} {local_path}"
        os.system(cmd)
        print("Done.")

if __name__ == "__main__":
    main()
