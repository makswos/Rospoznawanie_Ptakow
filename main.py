import os
import cv2
from ultralytics import YOLO

INPUT_FOLDER = r"C:\Users\mwos2\Documents\Projekt\Zdjecia" 
OUTPUT_FOLDER = r"C:\Users\mwos2\Documents\Projekt\Zdjecia_V2"    


os.makedirs(OUTPUT_FOLDER, exist_ok=True)


model = YOLO("yolov8s.pt")

BIRD_CLASS_ID = 14

print("Rozpoczynam przetwarzanie zdjęć...")

for file_name in os.listdir(INPUT_FOLDER):
    if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.heic')):
        continue
        
    image_path = os.path.join(INPUT_FOLDER, file_name)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Nie można wczytać pliku: {file_name}")
        continue

    results = model(image, conf=0.25, verbose=False)[0]
    
    bird_counter = 0

    if results.boxes is None:
        continue
    for box in results.boxes:
        cls_id = int(box.cls[0]) 
        
    
        if cls_id == BIRD_CLASS_ID:
            bird_counter += 1
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            height, width, _ = image.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            
            cropped_bird = image[y1:y2, x1:x2]
            
            if cropped_bird.size == 0:
                continue
                
            name_without_ext = os.path.splitext(file_name)[0]
            output_file_name = f"{name_without_ext}_ptak_{bird_counter}.jpg"
            output_path = os.path.join(OUTPUT_FOLDER, output_file_name)
            
            cv2.imwrite(output_path, cropped_bird)
            print(f"Zapisano: {output_file_name}")

print("\nPrzetwarzanie zakończone! Wycięte ptaki znajdziesz w folderze wyjściowym.")