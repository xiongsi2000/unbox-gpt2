import torch
from pathlib import Path
from transformer_lib.normalization import RMSNorm

FIXTURES_PATH = Path(__file__).parent / "fixtures"

def main():
    reference_weights = torch.load(FIXTURES_PATH / "rmsnorm_weights.pt")
    in_features = torch.load(FIXTURES_PATH / "in_features.pt")
    weight_tensor = reference_weights['weight']
    d_model = weight_tensor.shape[0]
    norm = RMSNorm(d_model, epsilon=1e-5)
    with torch.no_grad():
        norm.gains.data = weight_tensor
    output = norm(in_features)
    torch.save(output, FIXTURES_PATH / "rmsnorm_expected_output.pt")
    print("New reference output saved to rmsnorm_expected_output.pt")

if __name__ == "__main__":
    main() 