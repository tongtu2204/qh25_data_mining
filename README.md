# Stroke Risk Prediction — XGBoost + PCA + SHAP

Project tái hiện quy trình dự đoán nguy cơ đột quỵ của Mochurad et al. (2025), kết hợp:

- XGBoost;
- PCA giữ tối thiểu 95% phương sai;
- xử lý mất cân bằng lớp;
- Grid Search;
- Cross-validation;
- SHAP/XAI;
- kiểm định thống kê;
- Jacobi eigendecomposition bằng C++;
- song song hóa Jacobi bằng OpenMP.

Project chạy trên hai bộ dữ liệu:

1. `healthcare-dataset-stroke-data.csv` — **Dataset 1**.
2. `healthcare_data_2GB.csv.zip` — **Dataset 2 synthetic**.

Dataset 2 được đọc trực tiếp từ file ZIP, không cần giải nén thủ công.

---

# 1. Cấu trúc project

```text
qh25_data_mining/
│
├── .gitignore
├── latexmkrc
├── README.md
├── requirements.txt
│
├── .vscode/
│   └── settings.json
│
├── config/
│   ├── config.example.yaml
│   └── config.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── experiments/
│   ├── 01_data_check.py
│   ├── 02_prepare_data.py
│   ├── 03_pca_analysis.py
│   ├── 04_xgboost_baseline.py
│   ├── 05_hyperparameter_search.py
│   ├── 06_model_evaluation.py
│   ├── 07_shap_analysis.py
│   ├── 08_statistical_tests.py
│   ├── 09_pca_benchmark.py
│   ├── 10_final_reproduction.py
│   └── run_all.py
│
├── src/
│   ├── __init__.py
│   ├── balancing.py
│   ├── config.py
│   ├── evaluation.py
│   ├── explainability.py
│   ├── modeling.py
│   ├── pca.py
│   ├── plots.py
│   ├── preprocessing.py
│   ├── run_experiment.py
│   ├── statistics.py
│   ├── tuning.py
│   ├── utils.py
│   └── xgboost_model.py
│
├── cpp/
│   ├── CMakeLists.txt
│   ├── jacobi_serial.cpp
│   ├── jacobi_openmp.cpp
│   └── build/
│
├── scripts/
│   ├── check_data.py
│   ├── download_data.py
│   ├── prepare_npz.py
│   └── run_all.py
│
├── results/
│   ├── dataset1/
│   ├── dataset2/
│   ├── pca_benchmark/
│   └── final_reproduction/
│
├── report/
│   ├── main.tex
│   ├── Tomtat.tex
│   ├── Viettat.tex
│   ├── ThongtinBaibao.tex
│   ├── NguonBaibao.tex
│   ├── Chuong1_Boicanh.tex
│   ├── Chuong2_Phuongphap.tex
│   ├── Chuong3_Ketqua.tex
│   ├── Chuong4_Thaoluan.tex
│   ├── Ketluan.tex
│   ├── TaiLieuThamKhao.tex
│   └── figures_original/
│
└── slides/
    └── main.tex
```

## 1.1. Ý nghĩa từng thư mục

### `config/`

Chứa cấu hình phụ thuộc vào từng máy.

- `config.example.yaml`: file mẫu được lưu trên Git.
- `config.yaml`: cấu hình thực tế trên máy local, đặc biệt là đường dẫn Dataset 1 và Dataset 2. File này **không push lên Git**.

---

### `data/`

Dùng cho dữ liệu local.

- `raw/`: có thể đặt dữ liệu raw tại đây nếu muốn. Không bắt buộc vì project hỗ trợ đọc dữ liệu từ path bên ngoài.
- `processed/`: chứa các file `.npz` đã preprocessing/PCA để tái sử dụng nhanh ở các bước sau.

Các file dữ liệu lớn và `.npz` không được lưu trên Git.

---

### `experiments/`

Đây là **workflow thực nghiệm chính của project**.

Các script được đánh số theo đúng thứ tự chạy:

| File | Chức năng |
|---|---|
| `01_data_check.py` | Kiểm tra kích thước dữ liệu, missing, duplicate, phân phối target và outlier |
| `02_prepare_data.py` | Train/test split, xử lý missing, IQR clipping, encoding và chuẩn hóa |
| `03_pca_analysis.py` | PCA giữ ≥95% phương sai, phân tích explained variance và loadings |
| `04_xgboost_baseline.py` | So sánh baseline XGBoost có PCA và không PCA |
| `05_hyperparameter_search.py` | Cân bằng dữ liệu, PCA và Grid Search XGBoost |
| `06_model_evaluation.py` | Đánh giá final model và cross-validation |
| `07_shap_analysis.py` | Phân tích SHAP trên PCA space và projection về feature gốc |
| `08_statistical_tests.py` | So sánh PCA / không PCA bằng kiểm định thống kê |
| `09_pca_benchmark.py` | Benchmark Jacobi serial và OpenMP C++ |
| `10_final_reproduction.py` | Tổng hợp toàn bộ kết quả reproduction |
| `run_all.py` | Chạy tuần tự pipeline 01 → 10 |

Trong project hiện tại, `experiments/run_all.py` là runner chính để tái hiện toàn bộ thí nghiệm.

---

### `src/`

Chứa các module Python được dùng lại bởi các experiment.

- `config.py`: đọc `config.yaml` và quản lý đường dẫn.
- `preprocessing.py`: preprocessing dữ liệu.
- `balancing.py`: SMOTE / RandomUnderSampler.
- `pca.py`: PCA và các hàm liên quan.
- `xgboost_model.py`: cấu hình và train XGBoost.
- `tuning.py`: Grid Search và pipeline tuning.
- `evaluation.py`: tính Accuracy, Precision, Recall, F1, ROC-AUC, MCC, Cohen's Kappa...
- `explainability.py`: SHAP và feature importance.
- `statistics.py`: kiểm định thống kê.
- `plots.py`: các hàm vẽ biểu đồ.
- `utils.py`: các hàm hỗ trợ chung.
- `modeling.py`: các tiện ích liên quan tới modeling.
- `run_experiment.py`: entry point cho workflow model dạng module.

Nguyên tắc của project là:

```text
experiments/
    gọi
      ↓
src/
    xử lý logic dùng lại
```

---

### `cpp/`

Chứa phần thực nghiệm PCA/Jacobi bằng C++.

- `jacobi_serial.cpp`: Jacobi eigendecomposition bản tuần tự.
- `jacobi_openmp.cpp`: Jacobi eigendecomposition song song bằng OpenMP.
- `CMakeLists.txt`: cấu hình build C++.
- `build/`: file sinh ra bởi CMake/Ninja, không cần push Git.

Hai executable sau khi build:

```text
cpp/build/jacobi_serial.exe
cpp/build/jacobi_openmp.exe
```

được `experiments/09_pca_benchmark.py` gọi để benchmark.

---

### `scripts/`

Các utility script hỗ trợ.

- `check_data.py`: kiểm tra nhanh raw data.
- `download_data.py`: hỗ trợ chuẩn bị/download dữ liệu.
- `prepare_npz.py`: tạo cache NPZ.
- `run_all.py`: runner cũ/helper.

Workflow reproduction chính hiện tại nên sử dụng:

```powershell
python -m experiments.run_all
```

---

### `results/`

Chứa output thực nghiệm.

```text
results/
├── dataset1/
├── dataset2/
├── pca_benchmark/
└── final_reproduction/
```

Mỗi step có thể sinh:

- CSV;
- JSON;
- confusion matrix;
- ROC curve;
- PCA plots;
- SHAP plots;
- statistical test;
- benchmark OpenMP.

Các file kết quả cụ thể không được liệt kê trong README vì chúng được sinh lại từ experiment.

---

### `report/`

Báo cáo LaTeX dạng report.

`main.tex` là file chính và gọi các chương:

```text
Tomtat.tex
ThongtinBaibao.tex
Chuong1_Boicanh.tex
Chuong2_Phuongphap.tex
Chuong3_Ketqua.tex
Chuong4_Thaoluan.tex
Ketluan.tex
TaiLieuThamKhao.tex
```

`figures_original/` chứa hình lấy từ bài báo gốc.

---

### `slides/`

Chứa slide thuyết trình LaTeX.

File chính:

```text
slides/main.tex
```

---

# 2. Flow chạy project

Pipeline reproduction đầy đủ:

```text
Raw Dataset
     │
     ▼
┌─────────────────────────────┐
│ 01. Data Check              │
│ missing / outlier / target  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 02. Prepare Data            │
│ split / impute / IQR        │
│ scaling / one-hot           │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 03. PCA Analysis            │
│ retain >= 95% variance      │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 04. XGBoost Baseline        │
│ PCA vs no PCA               │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 05. Hyperparameter Search   │
│ balancing + PCA + GridSearch│
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 06. Model Evaluation        │
│ Test + Cross-validation     │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 07. SHAP / XAI              │
│ PC SHAP + projected feature │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 08. Statistical Tests       │
│ PCA vs no-PCA               │
└──────────────┬──────────────┘
               │
               ├────────────────────────┐
               │                        │
               ▼                        ▼
┌───────────────────────┐     ┌────────────────────────┐
│ Final ML results      │     │ 09. C++ Jacobi/OpenMP  │
└───────────┬───────────┘     │ benchmark              │
            │                 └────────────┬───────────┘
            │                              │
            └──────────────┬───────────────┘
                           ▼
              ┌──────────────────────────┐
              │ 10. Final Reproduction   │
              │ summary + paper compare  │
              └──────────────────────────┘
```

## Dataset 1

```text
Raw CSV
  ↓
Train/Test split
  ↓
BMI mean imputation
  ↓
IQR clipping
  ↓
Scaling + One-hot
  ↓
SMOTE trên train
  ↓
PCA 95%
  ↓
XGBoost
```

## Dataset 2

```text
ZIP
  ↓
Đọc CSV trực tiếp từ ZIP
  ↓
Binary threshold
  ↓
Drop missing
  ↓
Train/Test split
  ↓
IQR clipping
  ↓
RandomUnderSampler trên train
  ↓
Scaling + One-hot
  ↓
PCA 95%
  ↓
XGBoost
```

Test set không được balancing.

---

# 3. Cài đặt Python

Khuyến nghị Python 3.12.

Mở PowerShell tại thư mục project:

```powershell
python -m venv .venv
```

Activate:

```powershell
.venv\Scripts\Activate.ps1
```

Cập nhật pip:

```powershell
python -m pip install --upgrade pip
```

Cài dependencies:

```powershell
pip install -r requirements.txt
```

---

# 4. Cấu hình đường dẫn dữ liệu

Sau khi clone repo, copy:

```text
config/config.example.yaml
```

thành:

```text
config/config.yaml
```

Sau đó chỉ cần mở:

```text
config/config.yaml
```

và sửa đường dẫn tới hai dataset.

Ví dụ:

```yaml
data_raw_source:
  dataset1: 'C:/Users/Admin/Desktop/data/healthcare-dataset-stroke-data.csv'
  dataset2: 'C:/Users/Admin/Desktop/data/healthcare_data_2GB.csv.zip'
```

Nên dùng dấu `/` trong Windows path.

Không cần sửa path trong bất kỳ file Python nào.

`config/config.yaml` là cấu hình riêng của từng máy và đã được ignore khỏi Git.

---

# 5. Kiểm tra raw data

Dataset 1:

```powershell
python scripts/check_data.py --dataset dataset1 --nrows 10000
```

Dataset 2:

```powershell
python scripts/check_data.py --dataset dataset2 --nrows 10000
```

Dataset 2 được đọc trực tiếp từ ZIP:

```python
pd.read_csv(
    path,
    compression="zip",
)
```

Không cần giải nén file CSV 2GB ra ổ cứng.

---

# 6. Chạy từng experiment

Tất cả command nên chạy từ root:

```text
qh25_data_mining/
```

## Step 01 — Data check

```powershell
python -m experiments.01_data_check --dataset dataset1
python -m experiments.01_data_check --dataset dataset2
```

## Step 02 — Prepare data

```powershell
python -m experiments.02_prepare_data --dataset dataset1
python -m experiments.02_prepare_data --dataset dataset2
```

## Step 03 — PCA

```powershell
python -m experiments.03_pca_analysis --dataset dataset1
python -m experiments.03_pca_analysis --dataset dataset2
```

## Step 04 — XGBoost baseline

```powershell
python -m experiments.04_xgboost_baseline --dataset dataset1
python -m experiments.04_xgboost_baseline --dataset dataset2
```

## Step 05 — Balancing + tuning

```powershell
python -m experiments.05_hyperparameter_search --dataset dataset1
```

Dataset 2 sử dụng subset cho Grid Search để giảm thời gian:

```powershell
python -m experiments.05_hyperparameter_search --dataset dataset2 --max-tuning-rows 150000
```

## Step 06 — Model evaluation

```powershell
python -m experiments.06_model_evaluation --dataset dataset1 --cv 10
python -m experiments.06_model_evaluation --dataset dataset2 --cv 10
```

## Step 07 — SHAP

```powershell
python -m experiments.07_shap_analysis --dataset dataset1
python -m experiments.07_shap_analysis --dataset dataset2
```

## Step 08 — Statistical tests

```powershell
python -m experiments.08_statistical_tests --dataset dataset1 --cv 10 --n-jobs 2
python -m experiments.08_statistical_tests --dataset dataset2 --cv 10 --n-jobs 2
```

## Step 09 — Jacobi/OpenMP benchmark

Yêu cầu đã build C++ trước.

```powershell
python -m experiments.09_pca_benchmark --sizes 100 200 300 400 500 --repeats 3 --threads 2 --max-sweeps 20
```

## Step 10 — Final reproduction

```powershell
python -m experiments.10_final_reproduction
```

---

# 7. Chạy toàn bộ pipeline

Sau khi:

1. cài Python dependencies;
2. sửa `config/config.yaml`;
3. build C++ OpenMP;

có thể chạy toàn bộ reproduction bằng:

```powershell
python -m experiments.run_all
```

Flow:

```text
experiments.run_all
       │
       ├── Dataset 1
       │    └── Step 01 → 08
       │
       ├── Dataset 2
       │    └── Step 01 → 08
       │
       ├── Step 09 OpenMP
       │
       └── Step 10 Final Summary
```

Chỉ chạy Dataset 1:

```powershell
python -m experiments.run_all --dataset dataset1 --skip-openmp
```

Chỉ chạy Dataset 2:

```powershell
python -m experiments.run_all --dataset dataset2 --skip-openmp
```

Bỏ benchmark C++:

```powershell
python -m experiments.run_all --skip-openmp
```

Dataset 2 khá lớn nên full pipeline, đặc biệt Grid Search và Cross-validation, có thể chạy lâu.

---

# 8. Cache dữ liệu NPZ

Có thể tạo cache local để tránh preprocessing lại nhiều lần.

Dataset 1:

```powershell
python scripts/prepare_npz.py --dataset dataset1
```

Dataset 2:

```powershell
python scripts/prepare_npz.py --dataset dataset2
```

NPZ có thể chứa:

```text
X_train
X_test
y_train
y_test
feature_names
```

Các file `.npz` nằm trong:

```text
data/processed/
```

và không push lên Git.

---

# 9. Cài C++ / OpenMP trên Windows

Step 09 cần:

- GCC/G++;
- OpenMP;
- CMake;
- Ninja.

Project hiện sử dụng **MSYS2 UCRT64**.

## 9.1. Cài MSYS2

Cài MSYS2, sau đó mở:

```text
MSYS2 UCRT64
```

Cập nhật package:

```bash
pacman -Syu
```

Nếu MSYS2 yêu cầu đóng terminal sau lần update đầu tiên, mở lại `MSYS2 UCRT64` rồi chạy:

```bash
pacman -Syu
```

---

## 9.2. Cài GCC, CMake và Ninja

Trong terminal `MSYS2 UCRT64`:

```bash
pacman -S --needed \
mingw-w64-ucrt-x86_64-gcc \
mingw-w64-ucrt-x86_64-cmake \
mingw-w64-ucrt-x86_64-ninja
```

GCC package đã bao gồm hỗ trợ OpenMP.

---

## 9.3. Thêm MSYS2 vào Windows PATH

Thêm:

```text
C:\msys64\ucrt64\bin
```

vào:

```text
Environment Variables
→ User PATH
```

hoặc `System PATH`.

Sau khi thay PATH, đóng PowerShell/VS Code đang mở và mở lại.

Kiểm tra:

```powershell
g++ --version
cmake --version
ninja --version
```

Nếu cả ba command trả về version thì môi trường C++ đã sẵn sàng.

---

# 10. Build Jacobi C++

Tại root project:

```powershell
cmake -S cpp -B cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release
```

Build:

```powershell
cmake --build cpp/build
```

Kiểm tra:

```powershell
Get-ChildItem cpp\build -Filter "jacobi_*.exe"
```

Phải có:

```text
jacobi_serial.exe
jacobi_openmp.exe
```

Nếu đã build trước đó và chỉ sửa `.cpp`, chỉ cần:

```powershell
cmake --build cpp/build
```

không cần configure lại CMake.

Nếu di chuyển project sang đường dẫn mới hoặc đổi compiler/toolchain, nên xóa cache cũ:

```powershell
Remove-Item cpp\build -Recurse -Force -ErrorAction SilentlyContinue
```

rồi configure/build lại.

Khuyến nghị đường dẫn project không chứa ký tự Unicode đặc biệt để tránh vấn đề tương thích với một số toolchain Windows/MinGW.

---

# 11. Benchmark OpenMP

Benchmark mặc định:

```powershell
python -m experiments.09_pca_benchmark \
    --sizes 100 200 300 400 500 \
    --repeats 3 \
    --threads 2 \
    --max-sweeps 20
```

PowerShell có thể viết một dòng:

```powershell
python -m experiments.09_pca_benchmark --sizes 100 200 300 400 500 --repeats 3 --threads 2 --max-sweeps 20
```

Script so sánh:

```text
Jacobi Serial
      vs
Jacobi OpenMP
```

theo:

- execution time;
- speedup;
- relative eigenvalue error;
- orthogonality error.

---

# 12. Output

Output chính:

```text
results/
│
├── dataset1/
│   ├── 01_data_check/
│   ├── 02_prepare_data/
│   ├── 03_pca_analysis/
│   ├── 04_xgboost_baseline/
│   ├── 05_hyperparameter_search/
│   ├── 06_model_evaluation/
│   ├── 07_shap_analysis/
│   └── 08_statistical_tests/
│
├── dataset2/
│   └── ...
│
├── pca_benchmark/
│
└── final_reproduction/
```

`final_reproduction/` tổng hợp các kết quả cần thiết để so sánh reproduction với bài báo.

---

# 13. Git workflow

Các file local/data/build không nên commit:

```text
config/config.yaml
data raw
*.csv raw dataset
*.zip
*.npz
cpp/build/
__pycache__/
```

Workflow thông thường:

```text
          GitHub
             ▲
             │ git push
             │
          main branch
             ▲
             │ git commit
             │
        Staging Area
             ▲
             │ git add
             │
        Working Tree
```

Kiểm tra thay đổi:

```powershell
git status
```

Stage file cần thiết:

```powershell
git add experiments/10_final_reproduction.py
```

Commit:

```powershell
git commit -m "Update reproduction pipeline"
```

Push:

```powershell
git push origin main
```

Không nhất thiết dùng:

```powershell
git add .
```

Nếu chỉ sửa một vài file, nên add chính xác file cần commit để tránh đưa nhầm data/build/output vào commit.

---

# 14. LaTeX

Report và slide sử dụng **XeLaTeX**.

## Report

```text
report/main.tex
```

## Slide

```text
slides/main.tex
```

Trong VS Code + LaTeX Workshop chọn recipe XeLaTeX được cấu hình cho project.

---

# 15. Quy trình nhanh cho máy mới

Sau khi clone project:

```text
1. Clone repository
        ↓
2. Tạo Python venv
        ↓
3. pip install -r requirements.txt
        ↓
4. Copy config.example.yaml → config.yaml
        ↓
5. Sửa 2 đường dẫn dataset
        ↓
6. Cài MSYS2 UCRT64 + GCC + CMake + Ninja
        ↓
7. Build cpp/
        ↓
8. Kiểm tra Dataset 1 / Dataset 2
        ↓
9. python -m experiments.run_all
        ↓
10. Xem results/final_reproduction/
```

Các command chính:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt

cmake -S cpp -B cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build

python -m experiments.run_all
```

---

# Reference

Mochurad, L. et al. (2025).

**Improving stroke risk prediction by integrating XGBoost, optimized principal component analysis, and explainable artificial intelligence.**

BMC Medical Informatics and Decision Making, 25:63.