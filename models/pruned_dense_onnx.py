import torch
import torch.onnx
from foundations.hparams import ModelHparams
from models import registry


WEIGHTS_PTH = "/home/hice1/rpolaki3/scratch/rpolaki3/open_lth_data/lottery_fad1e89f2f31045fca3d2609b4b0355f/replicate_1/level_20/main/model_ep27_it0.pth"
OUT_ONNX = "conv2_pruned_dense.onnx"


state_dict = torch.load(WEIGHTS_PTH, map_location="cpu")

total = 0
nonzero = 0

for v in state_dict.values():
    total += v.numel()
    nonzero += torch.count_nonzero(v).item()

print("Total parameters:", total)
print("Non-zero parameters:", nonzero)
print("Sparsity (%):", round(100 * (1 - nonzero / total), 4))


model_hparams = ModelHparams(
    model_name="cifar_conv_2",
    model_init="kaiming_normal",
    batchnorm_init="uniform",
)

model = registry.get(model_hparams, outputs=10)
model.load_state_dict(state_dict)
model.eval()


dummy = torch.randn(1, 3, 32, 32)

torch.onnx.export(
    model,
    dummy,
    OUT_ONNX,
    opset_version=18,
    input_names=["input"],
    output_names=["logits"],
)

print("Exported:", OUT_ONNX)
