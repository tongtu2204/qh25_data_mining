import argparse
import subprocess
import sys
from pathlib import Path


def print_title(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def run_command(cmd):
    print()
    print("RUN:")
    print(" ".join(cmd))
    print()

    subprocess.run(
        cmd,
        check=True,
    )


def run_python_module(
    module,
    args=None,
):
    cmd = [
        sys.executable,
        "-m",
        module,
    ]

    if args:
        cmd.extend(args)

    run_command(cmd)


def check_cpp_executables():
    serial_candidates = [
        Path("cpp/build/jacobi_serial.exe"),
        Path("cpp/build/Release/jacobi_serial.exe"),
    ]

    openmp_candidates = [
        Path("cpp/build/jacobi_openmp.exe"),
        Path("cpp/build/Release/jacobi_openmp.exe"),
    ]

    serial_ok = any(
        path.exists()
        for path in serial_candidates
    )

    openmp_ok = any(
        path.exists()
        for path in openmp_candidates
    )

    return (
        serial_ok
        and openmp_ok
    )


def run_dataset_pipeline(
    dataset_name,
    dataset2_tuning_rows=150000,
    cv=10,
    n_jobs=2,
    shap_rows=2000,
):
    print_title(
        f"PIPELINE - {dataset_name.upper()}"
    )

    # ============================================================
    # 01 - DATA CHECK
    # ============================================================

    run_python_module(
        "experiments.01_data_check",
        [
            "--dataset",
            dataset_name,
        ],
    )

    # ============================================================
    # 02 - PREPARE DATA
    # ============================================================

    run_python_module(
        "experiments.02_prepare_data",
        [
            "--dataset",
            dataset_name,
        ],
    )

    # ============================================================
    # 03 - PCA ANALYSIS
    # ============================================================

    run_python_module(
        "experiments.03_pca_analysis",
        [
            "--dataset",
            dataset_name,
        ],
    )

    # ============================================================
    # 04 - XGBOOST BASELINE
    # ============================================================

    run_python_module(
        "experiments.04_xgboost_baseline",
        [
            "--dataset",
            dataset_name,
        ],
    )

    # ============================================================
    # 05 - BALANCING + PCA + TUNING
    # ============================================================

    step05_args = [
        "--dataset",
        dataset_name,
    ]

    if dataset_name == "dataset2":
        step05_args.extend([
            "--max-tuning-rows",
            str(
                dataset2_tuning_rows
            ),
        ])

    run_python_module(
        "experiments.05_hyperparameter_search",
        step05_args,
    )

    # ============================================================
    # 06 - MODEL EVALUATION
    # ============================================================

    run_python_module(
        "experiments.06_model_evaluation",
        [
            "--dataset",
            dataset_name,
            "--cv",
            str(cv),
        ],
    )

    # ============================================================
    # 07 - SHAP
    # ============================================================

    run_python_module(
        "experiments.07_shap_analysis",
        [
            "--dataset",
            dataset_name,
            "--max-rows",
            str(
                shap_rows
            ),
        ],
    )

    # ============================================================
    # 08 - STATISTICAL TESTS
    # ============================================================

    run_python_module(
        "experiments.08_statistical_tests",
        [
            "--dataset",
            dataset_name,
            "--cv",
            str(cv),
            "--n-jobs",
            str(n_jobs),
        ],
    )


def run_openmp_benchmark(
    threads=2,
):
    print_title(
        "09 - OPENMP PCA BENCHMARK"
    )

    if not check_cpp_executables():
        print(
            "SKIP STEP 09:"
        )

        print(
            "Không tìm thấy "
            "jacobi_serial.exe / "
            "jacobi_openmp.exe."
        )

        print(
            "Build trước bằng:"
        )

        print(
            "cmake -S cpp -B cpp/build "
            "-G Ninja "
            "-DCMAKE_BUILD_TYPE=Release"
        )

        print(
            "cmake --build cpp/build"
        )

        return

    run_python_module(
        "experiments.09_pca_benchmark",
        [
            "--sizes",
            "100",
            "200",
            "300",
            "400",
            "500",
            "--repeats",
            "3",
            "--threads",
            str(threads),
            "--max-sweeps",
            "20",
        ],
    )


def run_final_summary():
    print_title(
        "10 - FINAL REPRODUCTION SUMMARY"
    )

    run_python_module(
        "experiments.10_final_reproduction"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the full stroke prediction "
            "reproduction pipeline."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=[
            "dataset1",
            "dataset2",
            "all",
        ],
        default="all",
    )

    parser.add_argument(
        "--cv",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--dataset2-tuning-rows",
        type=int,
        default=150000,
    )

    parser.add_argument(
        "--shap-rows",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--openmp-threads",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--skip-openmp",
        action="store_true",
    )

    parser.add_argument(
        "--skip-final-summary",
        action="store_true",
    )

    args = parser.parse_args()

    print_title(
        "STROKE PREDICTION REPRODUCTION"
    )

    if args.dataset in [
        "dataset1",
        "all",
    ]:
        run_dataset_pipeline(
            dataset_name="dataset1",
            dataset2_tuning_rows=
                args.dataset2_tuning_rows,
            cv=args.cv,
            n_jobs=args.n_jobs,
            shap_rows=args.shap_rows,
        )

    if args.dataset in [
        "dataset2",
        "all",
    ]:
        run_dataset_pipeline(
            dataset_name="dataset2",
            dataset2_tuning_rows=
                args.dataset2_tuning_rows,
            cv=args.cv,
            n_jobs=args.n_jobs,
            shap_rows=args.shap_rows,
        )

    if not args.skip_openmp:
        run_openmp_benchmark(
            threads=
                args.openmp_threads
        )

    if not args.skip_final_summary:
        run_final_summary()

    print_title(
        "ALL REQUESTED STEPS COMPLETED"
    )


if __name__ == "__main__":
    main()