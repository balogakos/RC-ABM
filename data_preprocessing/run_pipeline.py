import os
import subprocess
import sys
import time

def run_script(script_path):
    """Runs a python script as a subprocess and monitors output."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {os.path.basename(script_path)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        # Using sys.executable to ensure we use the same environment
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Stream output to console in real-time
        for line in process.stdout:
            print(f"  {line.strip()}")
            sys.stdout.flush()

        process.wait()
        
        elapsed = time.time() - start_time
        if process.returncode == 0:
            print(f"---> SUCCESS (took {elapsed:.1f}s)")
            return True
        else:
            print(f"---> FAILED (Return code: {process.returncode})")
            return False
            
    except Exception as e:
        print(f"---> ERROR: {e}")
        return False

def main():
    # Detect directories
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define steps in order
    steps = [
        os.path.join(root_dir, "01_Retail_Centre_Processing", "assign_retail_centres.py"),
        os.path.join(root_dir, "01_Retail_Centre_Processing", "process_consumer_retail.py"),
        os.path.join(root_dir, "02_Network_Routing", "walking_network_code.py"),
        os.path.join(root_dir, "02_Network_Routing", "driving_network_code.py"),
        # os.path.join(root_dir, "02_Network_Routing", "pt_network_code.py"), # Optional - can be slow
        os.path.join(root_dir, "03_Travel_Time_Conversion", "distance_to_minutes.py"),
        os.path.join(root_dir, "04_Utility_Calculation", "prepare_data_for_utilities.py"),
        os.path.join(root_dir, "05_Visual_Validation", "validate_pipeline.py"),
    ]
    
    print("RETAIL ABM - DATA PREPROCESSING PIPELINE")
    print(f"Project Root: {root_dir}")
    print(f"Python: {sys.executable}\n")
    
    overall_start = time.time()
    
    for script in steps:
        if not os.path.exists(script):
            print(f"Skipping missing script: {script}")
            continue
            
        success = run_script(script)
        if not success:
            print("\n!!! PIPELINE ABORTED DUE TO ERROR !!!")
            sys.exit(1)
            
    total_time = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE - TOTAL TIME: {total_time/60:.1f} minutes")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
