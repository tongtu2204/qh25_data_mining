import argparse
import os
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def print_title(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def find_executable(name):
    candidates = [
        Path("cpp/build") / f"{name}.exe",
        Path("cpp/build/Release") / f"{name}.exe",
        Path("cpp/build") / name,
        Path("cpp/build/Release") / name,
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Không tìm thấy executable: {name}\n"
        "Hãy build cpp trước."
    )


def parse_output(text):
    patterns = {
        "seconds":
            r"seconds=([0-9eE+\-.]+)",
        "sweeps":
            r"sweeps=([0-9]+)",
        "max_offdiag":
            r"max_offdiag=([0-9eE+\-.]+)",
        "orthogonality_error":
            r"orthogonality_error=([0-9eE+\-.]+)",
        "threads":
            r"threads=([0-9]+)",
    }

    result = {}

    for key, pattern in patterns.items():
        match = re.search(
            pattern,
            text,
        )

        if match:
            result[key] = match.group(1)

    return result


def run_binary(
    executable,
    n,
    max_sweeps,
    tolerance,
    eigenvalue_path,
    threads=None,
):
    env = os.environ.copy()

    if threads is not None:
        env["OMP_NUM_THREADS"] = str(
            threads
        )

    cmd = [
        str(
            executable.resolve()
        ),
        str(n),
        str(max_sweeps),
        str(tolerance),
        str(
            eigenvalue_path.resolve()
        ),
    ]

    process = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )

    parsed = parse_output(
        process.stdout
    )

    return (
        parsed,
        process.stdout.strip(),
    )


def load_eigenvalues(path):
    return np.loadtxt(
        path,
        dtype=float,
    )


def relative_eigen_error(
    serial_values,
    openmp_values,
):
    denominator = max(
        np.linalg.norm(
            serial_values
        ),
        1e-15,
    )

    return float(
        np.linalg.norm(
            serial_values
            - openmp_values
        )
        / denominator
    )


def save_time_plot(
    df,
    output_path,
):
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        df["n"],
        df["serial_seconds"],
        marker="o",
        label="Serial",
    )

    ax.plot(
        df["n"],
        df["openmp_seconds"],
        marker="o",
        label="OpenMP",
    )

    ax.set_xlabel(
        "Matrix size n"
    )

    ax.set_ylabel(
        "Execution time (seconds)"
    )

    ax.set_title(
        "Jacobi eigendecomposition execution time"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_speedup_plot(
    df,
    output_path,
):
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        df["n"],
        df["speedup"],
        marker="o",
    )

    ax.axhline(
        y=1.0,
        linestyle="--",
    )

    ax.set_xlabel(
        "Matrix size n"
    )

    ax.set_ylabel(
        "Speedup (Serial / OpenMP)"
    )

    ax.set_title(
        "OpenMP speedup"
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def run_benchmark(
    sizes,
    repeats,
    threads,
    max_sweeps,
    tolerance,
):
    output_dir = (
        Path("results")
        / "pca_benchmark"
    )

    temp_dir = (
        output_dir
        / "eigenvalues"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    serial_exe = find_executable(
        "jacobi_serial"
    )

    openmp_exe = find_executable(
        "jacobi_openmp"
    )

    print_title(
        "JACOBI PCA BENCHMARK"
    )

    print(
        f"Serial executable : "
        f"{serial_exe}"
    )

    print(
        f"OpenMP executable : "
        f"{openmp_exe}"
    )

    print(
        f"Threads           : "
        f"{threads}"
    )

    print(
        f"Repeats           : "
        f"{repeats}"
    )

    print(
        f"Tolerance         : "
        f"{tolerance}"
    )

    print(
        f"Max sweeps        : "
        f"{max_sweeps}"
    )

    all_runs = []
    summary_rows = []

    for n in sizes:
        print_title(
            f"MATRIX {n} x {n}"
        )

        serial_times = []
        openmp_times = []

        serial_ortho = []
        openmp_ortho = []

        serial_offdiag = []
        openmp_offdiag = []

        eigen_errors = []

        serial_sweeps = []
        openmp_sweeps = []

        for repeat in range(
            1,
            repeats + 1,
        ):
            serial_path = (
                temp_dir
                / (
                    f"serial_n{n}_"
                    f"r{repeat}.txt"
                )
            )

            openmp_path = (
                temp_dir
                / (
                    f"openmp_n{n}_"
                    f"r{repeat}.txt"
                )
            )

            (
                serial_result,
                serial_text,
            ) = run_binary(
                executable=serial_exe,
                n=n,
                max_sweeps=max_sweeps,
                tolerance=tolerance,
                eigenvalue_path=serial_path,
            )

            (
                openmp_result,
                openmp_text,
            ) = run_binary(
                executable=openmp_exe,
                n=n,
                max_sweeps=max_sweeps,
                tolerance=tolerance,
                eigenvalue_path=openmp_path,
                threads=threads,
            )

            serial_values = (
                load_eigenvalues(
                    serial_path
                )
            )

            openmp_values = (
                load_eigenvalues(
                    openmp_path
                )
            )

            eig_error = (
                relative_eigen_error(
                    serial_values,
                    openmp_values,
                )
            )

            serial_time = float(
                serial_result[
                    "seconds"
                ]
            )

            openmp_time = float(
                openmp_result[
                    "seconds"
                ]
            )

            speedup = (
                serial_time
                / openmp_time
            )

            serial_times.append(
                serial_time
            )

            openmp_times.append(
                openmp_time
            )

            serial_ortho.append(
                float(
                    serial_result[
                        "orthogonality_error"
                    ]
                )
            )

            openmp_ortho.append(
                float(
                    openmp_result[
                        "orthogonality_error"
                    ]
                )
            )

            serial_offdiag.append(
                float(
                    serial_result[
                        "max_offdiag"
                    ]
                )
            )

            openmp_offdiag.append(
                float(
                    openmp_result[
                        "max_offdiag"
                    ]
                )
            )

            eigen_errors.append(
                eig_error
            )

            serial_sweeps.append(
                int(
                    serial_result[
                        "sweeps"
                    ]
                )
            )

            openmp_sweeps.append(
                int(
                    openmp_result[
                        "sweeps"
                    ]
                )
            )

            all_runs.append({
                "n":
                    n,
                "repeat":
                    repeat,
                "serial_seconds":
                    serial_time,
                "openmp_seconds":
                    openmp_time,
                "speedup":
                    speedup,
                "serial_sweeps":
                    serial_sweeps[-1],
                "openmp_sweeps":
                    openmp_sweeps[-1],
                "serial_max_offdiag":
                    serial_offdiag[-1],
                "openmp_max_offdiag":
                    openmp_offdiag[-1],
                "serial_orthogonality_error":
                    serial_ortho[-1],
                "openmp_orthogonality_error":
                    openmp_ortho[-1],
                "relative_eigenvalue_error":
                    eig_error,
            })

            print(
                f"Repeat {repeat}: "
                f"serial="
                f"{serial_time:.6f}s, "
                f"openmp="
                f"{openmp_time:.6f}s, "
                f"speedup="
                f"{speedup:.3f}x, "
                f"eig_error="
                f"{eig_error:.3e}"
            )

        serial_mean = float(
            np.mean(
                serial_times
            )
        )

        openmp_mean = float(
            np.mean(
                openmp_times
            )
        )

        summary_rows.append({
            "n":
                n,
            "serial_seconds":
                serial_mean,
            "openmp_seconds":
                openmp_mean,
            "speedup":
                serial_mean
                / openmp_mean,
            "serial_seconds_std":
                float(
                    np.std(
                        serial_times,
                        ddof=1,
                    )
                    if repeats > 1
                    else 0.0
                ),
            "openmp_seconds_std":
                float(
                    np.std(
                        openmp_times,
                        ddof=1,
                    )
                    if repeats > 1
                    else 0.0
                ),
            "serial_sweeps":
                float(
                    np.mean(
                        serial_sweeps
                    )
                ),
            "openmp_sweeps":
                float(
                    np.mean(
                        openmp_sweeps
                    )
                ),
            "serial_max_offdiag":
                float(
                    np.max(
                        serial_offdiag
                    )
                ),
            "openmp_max_offdiag":
                float(
                    np.max(
                        openmp_offdiag
                    )
                ),
            "serial_orthogonality_error":
                float(
                    np.max(
                        serial_ortho
                    )
                ),
            "openmp_orthogonality_error":
                float(
                    np.max(
                        openmp_ortho
                    )
                ),
            "relative_eigenvalue_error":
                float(
                    np.max(
                        eigen_errors
                    )
                ),
        })

    runs_df = pd.DataFrame(
        all_runs
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    runs_df.to_csv(
        output_dir
        / "benchmark_all_runs.csv",
        index=False,
    )

    summary_df.to_csv(
        output_dir
        / "benchmark_results.csv",
        index=False,
    )

    numerical_cols = [
        "n",
        "serial_sweeps",
        "openmp_sweeps",
        "serial_max_offdiag",
        "openmp_max_offdiag",
        "serial_orthogonality_error",
        "openmp_orthogonality_error",
        "relative_eigenvalue_error",
    ]

    summary_df[
        numerical_cols
    ].to_csv(
        output_dir
        / "numerical_accuracy.csv",
        index=False,
    )

    save_time_plot(
        df=summary_df,
        output_path=(
            output_dir
            / "execution_time.png"
        ),
    )

    save_speedup_plot(
        df=summary_df,
        output_path=(
            output_dir
            / "speedup.png"
        ),
    )

    print_title(
        "FINAL BENCHMARK SUMMARY"
    )

    print(
        summary_df[
            [
                "n",
                "serial_seconds",
                "openmp_seconds",
                "speedup",
                "relative_eigenvalue_error",
                "openmp_orthogonality_error",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved to: "
        f"{output_dir.resolve()}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "09 - Serial vs OpenMP "
            "Jacobi PCA benchmark"
        )
    )

    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[
            100,
            200,
            300,
            400,
            500,
        ],
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max-sweeps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-8,
    )

    args = parser.parse_args()

    run_benchmark(
        sizes=args.sizes,
        repeats=args.repeats,
        threads=args.threads,
        max_sweeps=args.max_sweeps,
        tolerance=args.tolerance,
    )


if __name__ == "__main__":
    main()