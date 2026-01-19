# Auteur : Ahmed Walid – Master SI - Sorbonne Université

import torch
import torchvision

print("[INFO] Création du modèle DeepLabV3+ (ResNet-101)…")
# Charger le modèle pré-entraîné sur COCO (il utilise les mêmes classes que Cityscapes)
weights = torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
model = torchvision.models.segmentation.deeplabv3_resnet101(weights=weights)
model.eval()

dummy = torch.randn(1, 3, 512, 1024)  # résolution Cityscapes

onnx_path = "dnn/deeplabv3plus_cityscapes_local.onnx"
torch.onnx.export(
    model, dummy, onnx_path,
    verbose=False, opset_version=11,
    input_names=["input"], output_names=["output"]
)
print(f"[] Modèle exporté en ONNX : {onnx_path}")
