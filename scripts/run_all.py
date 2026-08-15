import subprocess
import sys

for dataset in ["dataset1", "dataset2"]:
    cmd = [sys.executable, "-m", "src.run_experiment", "--dataset", dataset]
    if dataset == "dataset2":
        cmd.append("--skip-cv")
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)
