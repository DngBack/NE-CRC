# Sửa lỗi vLLM thiếu Python.h

## Vấn đề

vLLM lỗi khi khởi tạo vì thiếu Python development headers (`Python.h`):
```
fatal error: Python.h: No such file or directory
```

## Giải pháp đã triển khai

### 1. Automatic Fallback

Code giờ tự động fallback về transformers khi vLLM fail:

```python
# Trong create_llm_inference()
if use_vllm and VLLM_AVAILABLE:
    try:
        return VLLMInference(model_name, **kwargs)
    except Exception as e:
        logger.warning("vLLM failed, falling back to transformers")
        return TransformersInference(model_name, **kwargs)
```

### 2. Better Error Messages

Khi vLLM fail, code sẽ:
- Log error message chi tiết
- Đề xuất cách fix (cài python3-dev)
- Tự động fallback về transformers

## Cách sửa vĩnh viễn (Optional)

Nếu muốn fix vLLM để dùng được:

### Option 1: Cài Python development headers

```bash
# For Python 3.12
sudo apt-get update
sudo apt-get install python3.12-dev

# Or for general Python 3
sudo apt-get install python3-dev
```

### Option 2: Cài build tools (nếu cần)

```bash
sudo apt-get install build-essential
```

### Option 3: Sử dụng transformers (Recommended)

Code đã tự động fallback về transformers, nên bạn không cần làm gì cả. Transformers sẽ chạy được mà không cần compile.

## Expected Behavior

Sau khi sửa:
1. ✅ Code thử dùng vLLM trước
2. ✅ Nếu vLLM fail → tự động fallback về transformers
3. ✅ Pipeline tiếp tục chạy bình thường
4. ✅ Logging rõ ràng về lỗi và fallback

## Testing

Chạy lại pilot experiment:

```bash
./scripts/run_pilot_uv.sh
```

Bạn sẽ thấy:
- Warning về vLLM fail
- Message về fallback to transformers
- Pipeline tiếp tục chạy với transformers

## Performance Impact

- **vLLM**: Nhanh hơn, tốt cho batch inference
- **Transformers**: Chậm hơn một chút nhưng vẫn đủ dùng cho pilot experiments

Với model nhỏ (0.5B), sự khác biệt không đáng kể.

