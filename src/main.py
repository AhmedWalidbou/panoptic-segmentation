# ==============================================================
# PANOPTIC SEGMENTATION - Version finale, multi-images et stable
# Inspirée de Kirillov et al. (CVPR 2019)
# Auteur : Ahmed Walid BOUANZOUL – Master SI - Sorbonne Université
# ==============================================================

import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as ort
import glob

print(" Script démarré...")
AFFICHER_IMAGES = True

# -----------------------------------------------------------------
# ÉTAPE A : CONFIGURATION
# -----------------------------------------------------------------
#IMAGE_DIR = "images"
#RESULTS_DIR = "results"
#os.makedirs(RESULTS_DIR, exist_ok=True)

#MODEL_PATH = os.path.join("dnn", "frozen_inference_graph_coco.pb")
#CONFIG_PATH = os.path.join("dnn", "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
#DEEPLAB_ONNX = os.path.join("dnn", "deeplabv3plus_cityscapes_local.onnx")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "../data/images")

RESULTS_DIR = os.path.join(BASE_DIR, "../results")

MODEL_PATH = os.path.join(BASE_DIR, "../dnn/frozen_inference_graph_coco.pb")
CONFIG_PATH = os.path.join(BASE_DIR, "../dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
DEEPLAB_ONNX = os.path.join(BASE_DIR, "../dnn/deeplabv3plus_cityscapes_local.onnx")


CONF_THRESHOLD = 0.5

# -----------------------------------------------------------------
# CHARGEMENT DES MODÈLES
# -----------------------------------------------------------------
print("[INFO] Chargement du modèle Mask R-CNN (Things)...")
net_things = cv2.dnn.readNetFromTensorflow(MODEL_PATH, CONFIG_PATH)

print("[INFO] Chargement du modèle DeepLabV3+ (Cityscapes local)...")
if not os.path.exists(DEEPLAB_ONNX):
    raise FileNotFoundError(" Le modèle 'deeplabv3plus_cityscapes_local.onnx' est introuvable. "
                            "Exécute d'abord export_deeplab_cityscapes.py")
session = ort.InferenceSession(DEEPLAB_ONNX, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# -----------------------------------------------------------------
# FONCTION DE CALCUL PQ
# -----------------------------------------------------------------
def compute_pq(things_map, stuff_map):
    inter = np.logical_and(things_map > 0, stuff_map > 0).sum()
    union = np.logical_or(things_map > 0, stuff_map > 0).sum()
    return inter / union if union > 0 else 0

# -----------------------------------------------------------------
# TRAITEMENT DES IMAGES
# -----------------------------------------------------------------
images = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
print(f"[INFO] {len(images)} images trouvées dans '{IMAGE_DIR}'.")

pq_scores = []

for path in images:
    name = os.path.basename(path)
    print(f"\n[INFO] === Traitement de : {name} ===")
    img = cv2.imread(path)
    if img is None:
        print(f"[] Impossible de lire {name}")
        continue

    hauteur, largeur, _ = img.shape

    # ---------------- THINGS (Mask R-CNN) ----------------
    blob = cv2.dnn.blobFromImage(img, swapRB=True, crop=False)
    net_things.setInput(blob)
    (boites, masques) = net_things.forward(["detection_out_final", "detection_masks"])
    nombre_objets = boites.shape[2]
    instance_map = np.zeros((hauteur, largeur), dtype="uint32")
    instance_id_to_class_id = {}
    current_id = 1

    for i in range(nombre_objets):
        b = boites[0, 0, i]
        score = b[2]
        if score < CONF_THRESHOLD:
            continue
        cid = int(b[1])
        if cid >= masques.shape[2]:
            print(f"[attention] Classe {cid} ignorée (hors limites masques)")
            continue

        x, y, x2, y2 = int(b[3]*largeur), int(b[4]*hauteur), int(b[5]*largeur), int(b[6]*hauteur)
        x, y, x2, y2 = max(0, x), max(0, y), min(largeur-1, x2), min(hauteur-1, y2)
        mask = masques[0, i, cid]
        h, w = y2 - y, x2 - x
        if h <= 0 or w <= 0:
            continue

        mask = cv2.resize(mask, (w, h))
        mask_bool = mask > 0.5
        instance_map[y:y2, x:x2][mask_bool] = current_id
        instance_id_to_class_id[current_id] = cid
        current_id += 1

    # ---------------- STUFF (DeepLabV3+) ----------------
    BLOB_TAILLE = (1024, 512)
    img_resized = cv2.resize(img, BLOB_TAILLE)
    blob = cv2.dnn.blobFromImage(
        img_resized,
        scalefactor=1/127.5,
        mean=(127.5,127.5,127.5),
        swapRB=True,
        crop=False
    )
    outputs = session.run(None, {input_name: blob.astype(np.float32)})
    output_stuff = np.array(outputs[0])  # (1, classes, H, W)
    semantic_map_small = np.argmax(output_stuff[0], axis=0).astype("uint8")
    semantic_map = cv2.resize(semantic_map_small, (largeur, hauteur), interpolation=cv2.INTER_NEAREST)

    # ---------------- FUSION PANOPTIC ----------------
    panoptic_map = np.zeros((hauteur, largeur, 2), dtype=np.uint16)
    for inst_id, cid in instance_id_to_class_id.items():
        mask = instance_map == inst_id
        panoptic_map[mask, 0] = cid
        panoptic_map[mask, 1] = inst_id

    for y in range(hauteur):
        for x in range(largeur):
            if panoptic_map[y, x, 1] == 0:
                panoptic_map[y, x, 0] = semantic_map[y, x]

    # Visualisation
    output_panoptic = img.copy()
    color_map = {}
    for y in range(hauteur):
        for x in range(largeur):
            key = (int(panoptic_map[y, x, 0]), int(panoptic_map[y, x, 1]))
            if key not in color_map:
                color_map[key] = [random.randint(0, 255) for _ in range(3)]
            output_panoptic[y, x] = (
                0.4 * output_panoptic[y, x] + 0.6 * np.array(color_map[key])
            ).astype(np.uint8)

    save_path = os.path.join(RESULTS_DIR, name)
    cv2.imwrite(save_path, output_panoptic)
    pq = compute_pq(instance_map, semantic_map)
    pq_scores.append(pq)
    print(f"[ok] {name} traité — PQ = {pq:.3f} — Résultat sauvegardé : {save_path}")

# -----------------------------------------------------------------
# RÉSUMÉ FINAL
# -----------------------------------------------------------------
if pq_scores:
    pq_mean = np.mean(pq_scores)
    print("\n === RÉSUMÉ DES RÉSULTATS ===")
    for i, path in enumerate(images):
        print(f"   • {os.path.basename(path)}  →  PQ = {pq_scores[i]:.3f}")
    print(f"\n Moyenne PQ globale : {pq_mean:.3f}")
    print(f" Résultats enregistrés dans le dossier : {RESULTS_DIR}")
else:
    print("[attention] Aucune image n’a pu être traitée.")



if AFFICHER_IMAGES:
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(output_panoptic, cv2.COLOR_BGR2RGB))
    plt.title(f"Résultat Panoptique : {name}")
    plt.axis("off")
    plt.show()
