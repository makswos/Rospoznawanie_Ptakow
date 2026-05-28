"""
predict.py — testuje wytrenowany model na zdjęciach

Użycie:
  python predict.py                          # zdjęcia z folderu Zdjecia_V2
  python predict.py zdjecie.jpg              # konkretne zdjęcie
  python predict.py "C:/sciezka/do/folderu" # wszystkie zdjęcia z folderu
  python predict.py zdjecie.jpg --top 5     # top 5 zamiast domyślnych 3
  python predict.py zdjecie.jpg --no-yolo   # bez detekcji YOLO
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from ultralytics import YOLO

BASE         = Path(r"F:\CDV\SEZON 2\UCZENIE MASZYNOWE - PYTHON\Projekt Zaliczeniowy")
MODEL_PATH   = BASE / "bird_classifier.pt"
CLASSES_PATH = BASE / "bird_classes.json"
YOLO_PATH    = BASE / "yolov8s.pt"
DEFAULT_FOLDER = BASE / "Zdjecia_V2"

IMG_SIZE     = 224
YOLO_CONF    = 0.25
BIRD_CLASS   = 14
SUPPORTED    = {".png", ".jpg", ".jpeg", ".webp"}


def load_classifier(classes_path, model_path, device):
    with open(classes_path, encoding="utf-8") as f:
        classes = json.load(f)

    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.classifier[1].in_features, len(classes)),
    )
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, classes


def crop_bird_yolo(yolo_model, img_bgr):
    """
    Wykrywa ptaka na zdjęciu przez YOLO i zwraca przycięty fragment.
    Jeśli nie wykryto — zwraca oryginał.
    """
    results = yolo_model(img_bgr, conf=YOLO_CONF, verbose=False)[0]
    if results.boxes is None:
        return img_bgr, False

    best_box = None
    best_area = 0
    for box in results.boxes:
        if int(box.cls[0]) != BIRD_CLASS:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best_box  = (x1, y1, x2, y2)

    if best_box is None:
        return img_bgr, False

    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = best_box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return img_bgr[y1:y2, x1:x2], True


def predict_image(img_path: Path, classifier, classes, transform, device,
                  yolo_model=None, top_k=3):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"  [BŁĄD] Nie można otworzyć: {img_path.name}")
        return

    detected = False
    if yolo_model is not None:
        img_bgr, detected = crop_bird_yolo(yolo_model, img_bgr)

    # BGR → RGB → PIL
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    tensor  = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(classifier(tensor), dim=1)[0]

    top = torch.topk(probs, min(top_k, len(classes)))

    yolo_tag = "✓ YOLO" if detected else ("✗ brak ptaka" if yolo_model else "bez YOLO")
    print(f"\n{'─'*55}")
    print(f"  Zdjęcie : {img_path.name}  [{yolo_tag}]")
    print(f"{'─'*55}")
    for rank, (prob, idx) in enumerate(zip(top.values, top.indices), 1):
        name = classes[idx.item()].replace("_", " ").title()
        bar  = "█" * int(prob.item() * 30)
        print(f"  {rank}. {name:<38s}  {prob.item()*100:5.1f}%  {bar}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", nargs="?", default=None,
                        help="Ścieżka do zdjęcia lub folderu (domyślnie: Zdjecia_V2)")
    parser.add_argument("--top",     type=int, default=3,
                        help="Ile najlepszych wyników pokazać (domyślnie 3)")
    parser.add_argument("--no-yolo", action="store_true",
                        help="Nie używaj YOLO do detekcji ptaka")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie: {device}")

    print("Ładuję klasyfikator...", end=" ", flush=True)
    classifier, classes = load_classifier(CLASSES_PATH, MODEL_PATH, device)
    print(f"OK  ({len(classes)} gatunków)")

    yolo_model = None
    if not args.no_yolo and YOLO_PATH.exists():
        print("Ładuję YOLO...", end=" ", flush=True)
        yolo_model = YOLO(str(YOLO_PATH))
        print("OK")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Zbierz pliki do predykcji
    target = Path(args.target) if args.target else DEFAULT_FOLDER
    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted(f for f in target.iterdir() if f.suffix.lower() in SUPPORTED)
        if not files:
            print(f"Brak zdjęć w: {target}")
            return
        print(f"Znaleziono {len(files)} zdjęć w: {target}\n")
    else:
        print(f"Nie znaleziono: {target}")
        return

    for img_path in files:
        predict_image(img_path, classifier, classes, transform, device,
                      yolo_model=yolo_model, top_k=args.top)


if __name__ == "__main__":
    main()
