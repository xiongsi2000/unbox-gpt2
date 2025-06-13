import torch
from pathlib import Path

weights_path = Path(__file__).parent / "fixtures" / "transformer_lm_weights.pt"
data = torch.load(weights_path)

# 先重命名
renamed_data = {}
for k, v in data.items():
    if k.endswith('.gains'):
        new_k = k.replace('.gains', '.weight')
        renamed_data[new_k] = v
    else:
        renamed_data[k] = v

# 再移除所有 .gains 结尾的 key（如果还有残留）
final_data = {k: v for k, v in renamed_data.items() if not k.endswith('.gains')}

torch.save(final_data, weights_path)
print("所有 .gains 已重命名为 .weight，并移除多余 .gains") 