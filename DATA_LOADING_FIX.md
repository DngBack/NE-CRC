# Sửa lỗi tải dữ liệu AbstentionBench

## Vấn đề

HuggingFace không còn hỗ trợ `trust_remote_code` cho dataset scripts, dẫn đến lỗi khi tải AbstentionBench dataset.

## Giải pháp đã triển khai

### 1. Multiple Loading Methods

Code giờ thử nhiều phương pháp load dataset theo thứ tự:

**Method 1**: Load không có `trust_remote_code` (preferred)
```python
load_dataset("facebook/AbstentionBench", cache_dir=...)
```

**Method 2**: Load với `trust_remote_code` (fallback cho older datasets)
```python
load_dataset("facebook/AbstentionBench", trust_remote_code=True, ...)
```

**Method 3**: Thử alternative dataset name
```python
load_dataset("AbstentionBench/AbstentionBench", ...)
```

**Method 4**: Load từ files trực tiếp (parquet/json)
```python
# List files từ HuggingFace Hub
# Load từ parquet hoặc JSON files trực tiếp
```

### 2. Helper Methods

Đã thêm 2 helper methods:

- `_try_load_via_datasets()`: Thử các phương pháp load chuẩn
- `_try_load_from_files()`: Load từ files trực tiếp qua huggingface_hub

### 3. Error Handling

- Tất cả methods đều có try-except
- Logging chi tiết cho mỗi method
- Fallback tự động về synthetic data nếu tất cả methods fail

## Cách sử dụng

Code tự động thử các methods theo thứ tự:

```python
loader = create_default_loader()
samples = loader.load_dataset("unknown", use_real_data=True)
```

Nếu load thất bại, sẽ tự động fallback về synthetic data với warning message.

## Dependencies

Đã thêm `huggingface-hub>=0.19.0` vào `pyproject.toml` để hỗ trợ direct file loading.

## Testing

Để test data loading:

```bash
# Test với script
python scripts/test_data_loading.py

# Hoặc test trực tiếp
python -c "from src.data import create_default_loader; loader = create_default_loader(); samples = loader.load_dataset('unknown', max_samples=5, use_real_data=True); print(f'Loaded {len(samples)} samples')"
```

## Expected Behavior

1. **Nếu dataset load được**: Sẽ thấy log "✓ Successfully loaded..."
2. **Nếu dataset không load được**: Sẽ thấy warnings và fallback về synthetic data
3. **Synthetic data**: Vẫn hoạt động bình thường như trước

## Notes

- Dataset AbstentionBench có thể không available hoặc có cấu trúc khác
- Code sẽ tự động fallback về synthetic nếu cần
- Synthetic data vẫn đủ để test pipeline
- Để có dữ liệu thật, cần dataset được convert sang format chuẩn (parquet/json)

