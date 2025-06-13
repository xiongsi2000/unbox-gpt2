import torch
from pathlib import Path

# 权重文件路径
weights_path = Path(__file__).parent / "fixtures" / "transformer_lm_weights.pt"

# 加载原始权重
data = torch.load(weights_path)

# 替换所有 .gains 为 .weight
new_data = {}
for k, v in data.items():
    if k.endswith('.gains'):
        new_k = k.replace('.gains', '.weight')
    else:
        new_k = k
    new_data[new_k] = v

# 保存回原文件
torch.save(new_data, weights_path)
print("转换完成，所有 .gains 已替换为 .weight") 