# Hướng Dẫn Chạy Repo NE-CRC - Bắt Đầu Cơ Bản

## 📋 Tổng Quan

Repo này implement **Shift-Aware UniCR (S-UniCR)** - một phương pháp để đánh giá độ tin cậy và dự đoán có chọn lọc dưới distribution shift.

## 🚀 Bước 1: Cài Đặt

### Yêu Cầu Hệ Thống
- Python 3.10 trở lên
- GPU có CUDA (khuyến nghị, nhưng không bắt buộc)
- Package manager `uv`

### Cài Đặt Dependencies

```bash
# 1. Đảm bảo bạn đã cài uv (nếu chưa có)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Cài đặt tất cả dependencies
uv sync

# 3. Kích hoạt virtual environment
source .venv/bin/activate

# 4. Kiểm tra cài đặt
python -c "import torch; print('✓ Cài đặt thành công!')"
```

## 🎯 Bước 2: Chạy Thử Nghiệm Pilot (Cơ Bản Nhất)

Đây là cách nhanh nhất để xem hệ thống hoạt động. Chọn một trong ba cách phù hợp với setup của bạn:

### Cách 1: Sử dụng Conda Environment
```bash
# Chạy với conda environment (mặc định: ne-crc)
./scripts/run_pilot_conda.sh

# Hoặc chỉ định tên environment khác
CONDA_ENV_NAME=your_env_name ./scripts/run_pilot_conda.sh
```

> **Lưu ý**: Đảm bảo bạn đã tạo conda environment và cài đặt dependencies trước.

### Cách 2: Sử dụng uv (Khuyến nghị cho setup mới)
```bash
# Chạy thử nghiệm pilot với uv
./scripts/run_pilot_uv.sh
```

> **Lưu ý**: Script này tự động quản lý environment với `uv`, không cần activate thủ công. Sẽ tự động chạy `uv sync` nếu cần.

### Cách 3: Auto-detect (Tự động phát hiện)
```bash
# Script tự động phát hiện và sử dụng environment có sẵn
./scripts/run_pilot.sh
```

> **Lưu ý**: Script này sẽ tự động tìm và sử dụng `.venv`, conda environment đang active, hoặc fallback sang `uv`.

**Thời gian chạy dự kiến**: 10-30 phút (tùy GPU)

### Script này sẽ tự động:
1. ✅ Load dataset "unknown" từ AbstentionBench
2. ✅ Chạy LLM inference (sẽ cache sau lần đầu)
3. ✅ Train 5 biến thể hệ thống
4. ✅ Đánh giá metrics đầy đủ
5. ✅ Tạo figures và tables

## 📊 Bước 3: Xem Kết Quả

Sau khi chạy xong, kết quả sẽ nằm trong thư mục `outputs/`:

```bash
# Xem tóm tắt kết quả
cat outputs/pilot_*/results_summary.md

# Hoặc xem chi tiết
ls -la outputs/pilot_*/
```

### Cấu Trúc Kết Quả:
```
outputs/pilot_id_unknown/
├── results.pkl              # Dữ liệu thô
├── metrics.json             # Metrics dạng JSON
├── figures/                 # Các biểu đồ
│   ├── rc_curves.pdf
│   ├── coverage_at_risk.pdf
│   └── ...
└── tables/                  # Bảng LaTeX
    ├── table_main_results.tex
    └── results_summary.md   # ⭐ BẮT ĐẦU TỪ ĐÂY
```

## 🔧 Bước 4: Chạy Thử Nghiệm Tùy Chỉnh

Nếu muốn thử nghiệm với cấu hình khác:

### 4.1. Tạo File Config Mới

```bash
# Copy template
cp experiments/config_templates/pilot.yaml experiments/my_experiment.yaml

# Chỉnh sửa config
nano experiments/my_experiment.yaml
```

### 4.2. Các Tham Số Quan Trọng Trong Config

```yaml
# Dataset
dataset_names:
  - unknown  # hoặc: outdated, false_premise, underspecified, multi_hop

# Loại shift
shift_type: id  # id (không shift), mild, strong

# Model
model_name: "meta-llama/Llama-3.2-1B"  # Model nhỏ để test nhanh
num_samples: 5  # Số mẫu cho uncertainty estimation

# Risk level
alpha: 0.05  # 5% error target
delta: 0.05  # 5% outlier threshold
```

### 4.3. Chạy Thử Nghiệm

```bash
# Chạy với config tùy chỉnh
python scripts/run_experiment.py --config experiments/my_experiment.yaml

# Tạo visualizations
python scripts/generate_figures.py --results outputs/my_experiment/
python scripts/generate_tables.py --results outputs/my_experiment/
```

## 📚 Giải Thích Các Khái Niệm Cơ Bản

### Các Biến Thể Hệ Thống

1. **Heuristic**: Threshold đơn giản trên uncertainty (không calibration)
2. **UniCR**: Baseline chuẩn (CRC với giả định exchangeability)
3. **UniCR+Filter**: UniCR + SConU outlier detection
4. **UniCR+NE-CRC**: UniCR + non-exchangeable CRC
5. **S-UniCR**: Hệ thống đầy đủ (Filter + NE-CRC) ⭐

### Loại Distribution Shift

- **ID** (In-Distribution): Split ngẫu nhiên, không có shift
- **MILD**: Shift nhẹ (cross-topic hoặc difficulty-based)
- **STRONG**: Shift mạnh (temporal + domain shift) - khó nhất

### Metrics Quan Trọng

- **Coverage**: Tỷ lệ mẫu được trả lời (cao hơn = trả lời nhiều hơn)
- **Selective Risk**: Tỷ lệ lỗi trên các mẫu đã trả lời (thấp hơn = tốt hơn)
- **AURC**: Area Under Risk-Coverage curve (thấp hơn = tốt hơn)

## 🐛 Xử Lý Lỗi Thường Gặp

### Lỗi "CUDA out of memory"
```bash
# Giảm số samples trong config
num_samples: 3  # Thay vì 5 hoặc 10

# Hoặc dùng model nhỏ hơn
model_name: "meta-llama/Llama-3.2-1B"
```

### Lỗi "Module not found"
```bash
# Cài lại dependencies
uv sync --reinstall

# Kiểm tra environment
source .venv/bin/activate
python -c "import src; print('OK')"
```

### Chạy chậm ở lần đầu
- Lần đầu sẽ chạy LLM inference (sẽ cache sau đó)
- Có thể dùng dataset nhỏ hơn để test
- Kiểm tra GPU: `nvidia-smi`

## 🎓 Lộ Trình Học Tập

1. ✅ **Bắt đầu**: Chạy `./scripts/run_pilot.sh`
2. 📊 **Xem kết quả**: Đọc `outputs/pilot_*/results_summary.md`
3. 🔬 **Thử nghiệm**: Thay đổi config và chạy lại
4. 📈 **Benchmark đầy đủ**: Chạy `./scripts/run_full_benchmark.sh` (mất vài giờ)
5. 📝 **Sử dụng kết quả**: Dùng các bảng LaTeX trong paper

## 📖 Tài Liệu Tham Khảo

- **README.md**: Tổng quan dự án
- **GETTING_STARTED.md**: Hướng dẫn chi tiết (tiếng Anh)
- **EXPERIMENTS.md**: Hướng dẫn chạy thử nghiệm
- **IMPLEMENTATION_STATUS.md**: Chi tiết kỹ thuật

## 💡 Mẹo Tối Ưu Hiệu Suất

### Tăng Tốc Inference
```bash
# Sử dụng vLLM (tự động detect nếu đã cài)
uv add vllm
```

### Giảm Memory
- Giảm `num_samples` trong config
- Dùng model nhỏ hơn
- Enable quantization (sửa trong code)

## 🎯 Tóm Tắt Nhanh

```bash
# 1. Cài đặt (chọn một trong hai)
uv sync                      # Nếu dùng uv
# hoặc
conda env create -f environment.yml  # Nếu dùng conda (nếu có file)

# 2. Chạy pilot (chọn một trong ba cách)
./scripts/run_pilot_conda.sh  # Sử dụng conda
./scripts/run_pilot_uv.sh    # Sử dụng uv (khuyến nghị)
./scripts/run_pilot.sh       # Auto-detect

# 3. Xem kết quả
cat outputs/pilot_*/results_summary.md
```

**Chúc bạn thành công! 🚀**

