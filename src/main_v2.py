# ===============================================================
# PANOPTIC SEGMENTATION - VERSION 2 (Améliorée & Stable)
# DeepLabV3 ResNet101 (COCO) + Mask R-CNN (COCO)
# Auteur :Ahmed Walid - Master SI 
# ===============================================================

import os
import cv2
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import random

print(" Lancement de PanopticSeg V2...")

# ---------------------------------------------------------------
# CONFIGURATION GÉNÉRALE
# ---------------------------------------------------------------
IMAGE_DIR = "images"
RESULTS_DIR = "results_v2"
os.makedirs(RESULTS_DIR, exist_ok=True)

MASK_RCNN_PATH = os.path.join("dnn", "frozen_inference_graph_coco.pb")
MASK_RCNN_CFG = os.path.join("dnn", "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

CONF_THRESHOLD = 0.5
AFFICHER_IMAGES = True

# ---------------------------------------------------------------
# CHARGEMENT DES MODÈLES
# ---------------------------------------------------------------
print("[INFO] Chargement du modèle Mask R-CNN (Things)...")
net_things = cv2.dnn.readNetFromTensorflow(MASK_RCNN_PATH, MASK_RCNN_CFG)

print("[INFO] Chargement du modèle DeepLabV3 ResNet101 (COCO)...")
deeplab = torchvision.models.segmentation.deeplabv3_resnet101(
    weights=torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
).eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
deeplab.to(device)
print(f"[INFO] DeepLabV3 chargé sur {device}.")

# ---------------------------------------------------------------
# CLASSES COCO (simplifiées) + COULEURS FIXES
# ---------------------------------------------------------------
CLASSES = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "street sign",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]
COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype="uint8")

# ---------------------------------------------------------------
# FONCTION UTILITAIRE : CALCUL DU PANOPTIC QUALITY (PQ)
# ---------------------------------------------------------------
def compute_pq(instance_map, semantic_map):
    intersection = np.logical_and(instance_map > 0, semantic_map > 0).sum()
    union = np.logical_or(instance_map > 0, semantic_map > 0).sum()
    return intersection / union if union > 0 else 0

# ---------------------------------------------------------------
# TRAITEMENT DES IMAGES
# ---------------------------------------------------------------
pq_scores = []

for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        print(f" Format non supporté pour {img_name}, ignoré.")
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f" Impossible de lire {img_name}")
        continue

    print(f"\n[INFO] === Traitement de : {img_name} ===")
    H, W, _ = img.shape

    # -----------------------------------------------------------
    # ÉTAPE 1 : SEGMENTATION D'INSTANCE (MASK R-CNN)
    # -----------------------------------------------------------
    blob = cv2.dnn.blobFromImage(img, swapRB=True, crop=False)
    net_things.setInput(blob)
    (boxes, masks) = net_things.forward(["detection_out_final", "detection_masks"])

    instance_map = np.zeros((H, W), dtype="uint16")
    instance_to_class = {}
    current_id = 1

    for i in range(boxes.shape[2]):
        box = boxes[0, 0, i]
        score = box[2]
        if score < CONF_THRESHOLD:
            continue
        class_id = int(box[1])
        if class_id >= masks.shape[2]:
            continue

        x1, y1, x2, y2 = int(box[3]*W), int(box[4]*H), int(box[5]*W), int(box[6]*H)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W-1, x2), min(H-1, y2)

        mask = masks[0, i, class_id]
        mask = cv2.resize(mask, (x2 - x1, y2 - y1))
        mask_bool = mask > 0.5
        instance_map[y1:y2, x1:x2][mask_bool] = current_id
        instance_to_class[current_id] = class_id
        current_id += 1

    # -----------------------------------------------------------
    # ÉTAPE 2 : SEGMENTATION SÉMANTIQUE (DEEPLAB)
    # -----------------------------------------------------------
    img_tensor = torchvision.transforms.functional.to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = deeplab(img_tensor)["out"][0]
    semantic_map = torch.argmax(output, dim=0).cpu().numpy().astype("uint8")

    # -----------------------------------------------------------
    # ÉTAPE 3 : FUSION PANOPTIQUE AMÉLIORÉE
    # -----------------------------------------------------------
    print("[INFO] Fusion Panoptique intelligente...")

    panoptic = img.copy().astype("float32")

    for y in range(H):
        for x in range(W):
            if instance_map[y, x] > 0:
                cls = instance_to_class[instance_map[y, x]]
                color = np.array(COLORS[cls], dtype="float32")
                alpha = 0.55
            else:
                cls = semantic_map[y, x]
                color = np.array(COLORS[cls], dtype="float32")
                alpha = 0.35

            panoptic[y, x] = (1 - alpha) * panoptic[y, x] + alpha * color

    panoptic = np.clip(panoptic, 0, 255).astype("uint8")

    # -----------------------------------------------------------
    # ÉTAPE 4 : CALCUL DU PQ ET SAUVEGARDE
    # -----------------------------------------------------------
    pq = compute_pq(instance_map, semantic_map)
    pq_scores.append(pq)
    save_path = os.path.join(RESULTS_DIR, f"panoptic_{img_name}")
    cv2.imwrite(save_path, panoptic)
    print(f" {img_name} traité — PQ = {pq:.3f}")

    # -----------------------------------------------------------
    # AFFICHAGE (OPTIONNEL)
    # -----------------------------------------------------------
    if AFFICHER_IMAGES:
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(panoptic, cv2.COLOR_BGR2RGB))
        plt.title(f"Résultat Panoptique V2 : {img_name}")
        plt.axis("off")
        plt.show()

# ---------------------------------------------------------------
# RÉSUMÉ FINAL
# ---------------------------------------------------------------
if pq_scores:
    print("\n === RÉSUMÉ DES RÉSULTATS ===")
    for i, img_name in enumerate(os.listdir(IMAGE_DIR)):
        if i < len(pq_scores):
            print(f"   • {img_name}  →  PQ = {pq_scores[i]:.3f}")
    print(f"\n Moyenne PQ globale : {np.mean(pq_scores):.3f}")
else:
    print("[attention] Aucune image traitée.")
