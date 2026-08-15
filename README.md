# Stroke Risk Prediction — XGBoost + PCA + SHAP

Project tái hiện quy trình của Mochurad et al. (2025) trên hai bộ dữ liệu:

1. `healthcare-dataset-stroke-data.csv` — Dataset 1.
2. `healthcare_data_2GB.csv.zip` — Dataset 2 synthetic.

Dataset 2 được **đọc trực tiếp từ file ZIP**, không cần và không nên giải nén thủ công.

## 1. Mở project

Giải nén project và mở chính folder `qh25_data_mining` bằng VS Code.

```text
stroke_project_vntex/
├─ config/
│  ├─ config.example.yaml
│  └─ config.yaml              # path riêng trên máy, không push Git
├─ data/
│  ├─ raw/                     # không bắt buộc dùng nếu dữ liệu nằm ngoài project
│  └─ processed/               # cache .npz local, không push Git
├─ results/
├─ src/
├─ scripts/
├─ report/
├─ slides/
├─ requirements.txt
└─ README.md
```

## 2. Tạo môi trường Python

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Cấu hình đường dẫn dữ liệu

Toàn bộ path dữ liệu được tách khỏi code và đặt tại:

```text
config/config.yaml
```

Máy hiện tại đã được cấu hình:

```yaml
data_raw_source:
  dataset1: 'C:/Users/Admin/Desktop/THS/1. Học tập/25K2_Khai phá dữ liệu/healthcare-dataset-stroke-data.csv'
  dataset2: 'C:/Users/Admin/Desktop/THS/1. Học tập/25K2_Khai phá dữ liệu/healthcare_data_2GB.csv.zip'
```

Nếu người khác clone Git:

1. Copy `config/config.example.yaml` thành `config/config.yaml`.
2. Sửa `data_raw_source.dataset1` và `data_raw_source.dataset2` theo máy của họ.
3. Không cần sửa bất kỳ file Python nào.

`config/config.yaml` đã nằm trong `.gitignore`, vì vậy mỗi người có thể dùng path riêng.

### Lưu ý với Windows path

Nên dùng dấu `/` trong YAML:

```yaml
dataset1: 'C:/Users/Admin/Desktop/data.csv'
```

thay vì ghi đường dẫn bằng dấu `\` trong chuỗi double quote.

## 4. Kiểm tra dữ liệu trước khi chạy

Kiểm tra nhanh Dataset 1:

```powershell
python scripts/check_data.py --dataset dataset1 --nrows 10000
```

Kiểm tra nhanh Dataset 2 trực tiếp từ ZIP:

```powershell
python scripts/check_data.py --dataset dataset2 --nrows 10000
```

Lệnh Dataset 2 chỉ đọc mẫu từ CSV nằm trong ZIP, **không tạo file giải nén**.

## 5. Chạy trực tiếp từ dữ liệu raw

Dataset 1:

```powershell
python -m src.run_experiment --dataset dataset1
```

Dataset 2:

```powershell
python -m src.run_experiment --dataset dataset2 --skip-cv
```

Chạy cả hai:

```powershell
python scripts/run_all.py
```

Khi không truyền `--source`, code tự lấy path từ `config/config.yaml`.

Nếu cần tạm override path:

```powershell
python -m src.run_experiment --dataset dataset1 --source "D:/other/path/data.csv"
```

## 6. Dataset 2 được đọc thế nào?

`healthcare_data_2GB.csv.zip` chứa trực tiếp file CSV. Code sử dụng:

```python
pd.read_csv(path, compression="zip", ...)
```

Do đó:

```text
healthcare_data_2GB.csv.zip
        ↓
Pandas giải nén theo luồng khi đọc
        ↓
DataFrame
```

Không có bước:

```text
ZIP → giải nén ra healthcare_data_2GB.csv trên ổ cứng
```

Code cũng ép các dtype về `float32`, `int8` và `category` để giảm RAM cho Dataset 2.

## 7. Xử lý riêng cho hai dataset

### Dataset 1

- 5,110 bản ghi.
- Nhãn `stroke` vốn là 0/1.
- BMI thiếu được xử lý bằng mean imputation trong preprocessing.
- Cân bằng lớp mặc định: **SMOTE**.

Pipeline:

```text
raw CSV
→ split train/test
→ IQR từ train
→ imputation + scaling + one-hot
→ SMOTE trên train
→ PCA 95% (ở nhánh PCA)
→ XGBoost
```

### Dataset 2

- 5,769,190 bản ghi.
- Đọc trực tiếp từ ZIP.
- Các biến `stroke`, `hypertension`, `heart_disease` trong dữ liệu synthetic có thể là số gần 0/1; code threshold tại 0.5 để đưa về nhị phân.
- Các dòng missing được loại theo mô tả của bài báo.
- Cân bằng mặc định: **RandomUnderSampler**.

Để tiết kiệm RAM, undersampling Dataset 2 được thực hiện trước one-hot/scaling trong pipeline train.

```text
ZIP
→ đọc CSV trực tiếp
→ binary threshold
→ drop missing
→ split train/test
→ IQR từ train
→ undersampling train
→ scaling + one-hot
→ PCA 95% (ở nhánh PCA)
→ XGBoost
```

Test set không bị undersampling.

## 8. Cache train/test thành NPZ

NPZ là **tùy chọn**, dùng để lưu local dữ liệu train/test đã chuyển sang ma trận số, giúp các lần thử model sau không phải đọc và encode raw data lại.

Dataset 1:

```powershell
python scripts/prepare_npz.py --dataset dataset1
```

Dataset 2:

```powershell
python scripts/prepare_npz.py --dataset dataset2
```

Đường dẫn output lấy từ config:

```yaml
processed_data:
  dataset1: 'data/processed/dataset1_train_test.npz'
  dataset2: 'data/processed/dataset2_train_test.npz'
```

Trong NPZ gồm:

```text
X_train
X_test
y_train
y_test
feature_names
```

`X` lưu `float32`, `y` lưu `int8` và file dùng `np.savez_compressed`.

**NPZ cũng không push lên Git.** `.gitignore` đã bỏ qua `*.npz` và `data/processed/*`.

Lưu ý: tạo NPZ cho Dataset 2 vẫn cần đọc toàn bộ ZIP và biến đổi toàn bộ train/test nên sẽ tốn RAM/thời gian một lần. Chạy trực tiếp pipeline raw vẫn là lựa chọn mặc định.

## 9. Các nhánh thực nghiệm

Mỗi dataset chạy ba cấu hình:

- `unbalanced_no_pca`: XGBoost, không cân bằng, không PCA.
- `balanced_no_pca`: cân bằng lớp + XGBoost.
- `balanced_pca95`: cân bằng lớp + PCA giữ ≥95% phương sai + XGBoost.

Các chỉ số gồm:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- MCC
- Cohen's Kappa
- TN, FP, FN, TP
- thời gian fit
- số principal components và explained variance nếu có PCA

## 10. Output

Kết quả được lưu vào:

```text
results/dataset1/
results/dataset2/
```

Gồm metrics, classification report, confusion matrix, ROC curve, PCA variance/loadings và SHAP plots.

## 11. Git

Không push dữ liệu raw, ZIP 316 MB, NPZ hay path local lên Git.

Các file đã được ignore:

```gitignore
config/config.yaml
*.csv
*.zip
*.npz
data/raw/*
data/processed/*
```

Git chỉ giữ:

- source code;
- `config/config.example.yaml`;
- report/slide;
- README;
- các kết quả nhỏ nếu chủ động bỏ khỏi ignore.

## 12. LaTeX

Report và slide dùng **XeLaTeX**. Trong VS Code + LaTeX Workshop chọn recipe:

```text
vntex: XeLaTeX
```

Report:

```text
report/main.tex
```

Slide:

```text
slides/main.tex
```
