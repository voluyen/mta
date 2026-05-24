# Brainstorm — Fix tensor size mismatch trong span_finetune.py

## Problem
- Lỗi runtime tại `span_utils.py:67` khi train spancsd Qwen1.5 0.5B/1.8B:
  `RuntimeError: The size of tensor a (255) must match the size of tensor b (256) at non-singleton dimension 1`
- `attention_mask`: (B, 256), `offsets_mapping` re-tokenize: (B, 255, 2) → mismatch khi mask AND.

## Root Cause
- `span_finetune.py:311-313`: re-tokenize decoded text với `padding=True` → chỉ pad tới longest re-tokenized seq trong batch, không pad tới SeqLen gốc của `input_ids`.
- Round-trip decode→tokenize Qwen1.5 (special token/whitespace handling) làm mất 1 token → seq ngắn hơn input_ids gốc.
- `span_utils.py:48-51` chỉ slice khi offsets_mapping > SeqLen, không pad khi < SeqLen.

## Bằng chứng
- `span_fdd_finetune.py:676-678` đã fix đúng pattern: `padding='max_length', max_length=model_batch['attention_mask'].shape[1], truncation=True`.
- Bug ở `span_finetune.py` là bị bỏ sót khi merge fix sang.

## Phương án (đã chọn: Option A)
Update `span_finetune.py:312-313`:

```python
offsets_mapping = tokenizer(input_texts, return_offsets_mapping=True,
                            padding='max_length',
                            max_length=model_batch['attention_mask'].shape[1],
                            truncation=True,
                            add_special_tokens=False, return_tensors='pt')['offset_mapping']
```

## Phương án bỏ qua
- B (defensive ở `span_utils.py`): pad offsets dummy có nguy cơ match span giả; phải dùng sentinel value lớn.
- C (cả A+B): an toàn nhất nhưng dư.

## Files affected
- `distillm-master/span_finetune.py` (line 312-313)

## Out of scope (cùng bug pattern, không fix lần này)
- `distillm-master/ablation_span_finetune.py:634`
- `distillm-master/span_finetune_ctkd.py:617`

## Validation
- Chạy lại `bash distillm-master/scripts/qwen1.5/spancsd/train_0.5B_1.8B_spancsd.sh` (hoặc entropy variant).
- Confirm step 0 chạy qua được `compute_overall_span_loss` không raise.

## Risks
- Truncation=True có thể cắt nếu re-tokenize ra dài hơn SeqLen. Khi đó các span cuối bị mất — chấp nhận được vì attention_mask vẫn align.

## Unresolved
- Có cần kiểm tra xem `spandistillm` (cũng dùng `span_finetune.py`) đã từng gặp bug này chưa? Có thể bug chỉ surface khi seq dài đúng max_length.
