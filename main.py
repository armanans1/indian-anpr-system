"""
Automatic Number Plate Recognition (ANPR) - The Sliding Window Engine
------------------------------------------------------------
Uses a Sliding Window algorithm to extract valid Indian plates 
from messy OCR strings that include noise like "IND" tags, 
plate screws (read as 'o'), and image watermarks.
"""

import cv2
import easyocr
import csv
import os
import re
from datetime import datetime

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
INPUT_SOURCE = "test_images"   
OUTPUT_CSV = "results.csv"
MIN_CONFIDENCE = 0.05
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

INDIAN_PLATE_REGEX = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$')

# Bi-Directional Correction Dictionaries
DIGIT_TO_LETTER = {
    '0': 'O', '1': 'I', '2': 'Z', '3': 'J', 
    '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': 'B'
}

LETTER_TO_DIGIT = {
    'O': '0', 'Q': '0', 'D': '0', 'U': '0', 'C': '0', 
    'I': '1', 'T': '1', 'J': '1', 
    'Z': '2', 
    'A': '4', 'L': '4', 
    'S': '5', 
    'G': '6', 
    'B': '8'
}

# ---------------------------------------------------------
# SETUP
# ---------------------------------------------------------
print("Loading AI Model... Please wait.")
reader = easyocr.Reader(["en"], gpu=False)

if not os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "plate_text", "confidence"])

def log_detection(plate_text, confidence):
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(timespec="seconds"), plate_text, round(confidence, 2)])

def extract_valid_plate(raw_text):
    """
    SLIDING WINDOW ALGORITHM:
    Strips noise, then slides a 9 or 10 character window across the 
    string to hunt for a valid plate hidden inside OCR garbage.
    """
    # Strip all spaces and symbols
    text = re.sub(r'[^A-Za-z0-9]', '', raw_text).upper()
    
    if len(text) < 8:
        return None

    # Slide a window of length 10, then length 9
    for window_size in [10, 9]:
        for i in range(len(text) - window_size + 1):
            substring = list(text[i : i + window_size])
            
            # 10-Character Format: SS DD LL DDDD
            if window_size == 10:
                for j in [0, 1, 4, 5]: 
                    if substring[j] in DIGIT_TO_LETTER: substring[j] = DIGIT_TO_LETTER[substring[j]]
                for j in [2, 3, 6, 7, 8, 9]: 
                    if substring[j] in LETTER_TO_DIGIT: substring[j] = LETTER_TO_DIGIT[substring[j]]
            
            # 9-Character Format: SS DD L DDDD
            elif window_size == 9:
                for j in [0, 1, 4]: 
                    if substring[j] in DIGIT_TO_LETTER: substring[j] = DIGIT_TO_LETTER[substring[j]]
                for j in [2, 3, 5, 6, 7, 8]: 
                    if substring[j] in LETTER_TO_DIGIT: substring[j] = LETTER_TO_DIGIT[substring[j]]
            
            candidate = "".join(substring)
            
            # If this specific window matches the Indian Plate format, extract it!
            if INDIAN_PLATE_REGEX.match(candidate):
                return candidate
                
    return None

def merge_ocr_results(ocr_results):
    if not ocr_results: return []
    valid = [res for res in ocr_results if res[2] >= MIN_CONFIDENCE]
    
    # Sort boxes by Y-coordinate to group into horizontal lines
    valid.sort(key=lambda x: (x[0][0][1] + x[0][2][1])/2) 
    
    lines = []
    for res in valid:
        bbox, text, conf = res
        cy = (bbox[0][1] + bbox[2][1]) / 2
        h = abs(bbox[2][1] - bbox[0][1])
        cx = (bbox[0][0] + bbox[2][0]) / 2
        
        placed = False
        for line in lines:
            avg_y = sum(i['cy'] for i in line) / len(line)
            # Extremely generous vertical grouping (handles slanted text and IND tags)
            if abs(cy - avg_y) < max(h * 1.5, 40):
                line.append({'bbox': bbox, 'text': text, 'conf': conf, 'cx': cx, 'cy': cy, 'h': h})
                placed = True
                break
        if not placed:
            lines.append([{'bbox': bbox, 'text': text, 'conf': conf, 'cx': cx, 'cy': cy, 'h': h}])

    candidates = []
    # Add individual blocks
    for res in valid:
        candidates.append((res[0], res[1], res[2]))
        
    # Add horizontally merged blocks
    for line in lines:
        if len(line) > 1:
            line.sort(key=lambda item: item['cx'])
            merged_text = "".join([i['text'] for i in line])
            avg_conf = sum(i['conf'] for i in line) / len(line)
            min_x = min(i['bbox'][0][0] for i in line)
            min_y = min(i['bbox'][0][1] for i in line)
            max_x = max(i['bbox'][2][0] for i in line)
            max_y = max(i['bbox'][2][1] for i in line)
            merged_bbox = ((min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y))
            candidates.append((merged_bbox, merged_text, avg_conf))

    return candidates

def process_frame(frame, save_to_csv=True):
    h, w = frame.shape[:2]
    
    # Upscale images to 1000px width. Makes blurry letters sharper for OCR.
    if w != 1000:
        scale = 1000 / w
        frame = cv2.resize(frame, (1000, int(h * scale)))

    raw_ocr = reader.readtext(frame)
    candidates = merge_ocr_results(raw_ocr)
    
    detected_plates = set()
    results = []

    for (bbox, text, confidence) in candidates:
        plate_text = extract_valid_plate(text)
        
        if plate_text and plate_text not in detected_plates:
            detected_plates.add(plate_text)
            results.append({"plate": plate_text, "confidence": confidence})
            
            if save_to_csv:
                log_detection(plate_text, confidence)
            
            (top_left, top_right, bottom_right, bottom_left) = bbox
            x, y = int(top_left[0]), int(top_left[1])
            x2, y2 = int(bottom_right[0]), int(bottom_right[1])

            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, plate_text, (x, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    return frame, results

def process_image_cli(img_path):
    frame = cv2.imread(img_path)
    if frame is None: return
    print(f"\n--- Scanning: {os.path.basename(img_path)} ---")
    
    annotated_frame, results = process_frame(frame)
    for res in results:
        print(f"  -> LOGGED: {res['plate']} (conf: {res['confidence']:.2f})")
    if not results:
        print("  -> No valid Indian plate detected.")

    cv2.imshow("ANPR Result (Press any key for next)", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_on_folder(folder_path):
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return
        
    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(IMAGE_EXTENSIONS)])
    for filename in image_files:
        process_image_cli(os.path.join(folder_path, filename))

if __name__ == "__main__":
    if os.path.isdir(INPUT_SOURCE):
        run_on_folder(INPUT_SOURCE)