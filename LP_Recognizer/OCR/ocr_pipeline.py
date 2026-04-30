import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import re
import easyocr
import numpy as np
import difflib

ALLOWLIST = "กขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ0123456789เแโใไาีืึุูะัิำ่้๊๋์็"
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
reader = easyocr.Reader(["th", "en"], gpu=True)

VALID_PROVINCES = [
    "กรุงเทพมหานคร", "กระบี่", "กาญจนบุรี", "กาฬสินธุ์", "กำแพงเพชร", "ขอนแก่น", 
    "จันทบุรี", "ฉะเชิงเทรา", "ชลบุรี", "ชัยนาท", "ชัยภูมิ", "ชุมพร", "เชียงราย", 
    "เชียงใหม่", "ตรัง", "ตราด", "ตาก", "นครนายก", "นครปฐม", "นครพนม", 
    "นครราชสีมา", "นครศรีธรรมราช", "นครสวรรค์", "นนทบุรี", "นราธิวาส", "น่าน", 
    "บึงกาฬ", "บุรีรัมย์", "ปทุมธานี", "ประจวบคีรีขันธ์", "ปราจีนบุรี", "ปัตตานี", 
    "พระนครศรีอยุธยา", "พะเยา", "พังงา", "พัทลุง", "พิจิตร", "พิษณุโลก", 
    "เพชรบุรี", "เพชรบูรณ์", "แพร่", "ภูเก็ต", "มหาสารคาม", "มุกดาหาร", 
    "แม่ฮ่องสอน", "ยโสธร", "ยะลา", "ร้อยเอ็ด", "ระนอง", "ระยอง", "ราชบุรี", 
    "ลพบุรี", "ลำปาง", "ลำพูน", "เลย", "ศรีสะเกษ", "สกลนคร", "สงขลา", "สตูล", 
    "สมุทรปราการ", "สมุทรสงคราม", "สมุทรสาคร", "สระแก้ว", "สระบุรี", "สิงห์บุรี", 
    "สุโขทัย", "สุพรรณบุรี", "สุราษฎร์ธานี", "สุรินทร์", "หนองคาย", "หนองบัวลำภู", 
    "อ่างทอง", "อำนาจเจริญ", "อุดรธานี", "อุตรดิตถ์", "อุทัยธานี", "อุบลราชธานี",
    "เบตง" # (Betong has its own license plates)
]

def preprocess_image(img):
    """
    Upscales and sharpens the image to help OCR read blurry text.
    """
    # 1. Upscale 2x using Cubic Interpolation
    img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    
    # 2. Grayscale to remove color noise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Bilateral Filter: Smooths background grain while keeping edges sharp
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 4. Sharpening Kernel: Makes the black text "pop" against the background
    kernel = np.array([[0, -1, 0], 
                       [-1, 5,-1], 
                       [0, -1, 0]])
    sharpened = cv2.filter2D(filtered, -1, kernel)
    
    # 5. Convert back to BGR because EasyOCR expects 3 color channels
    final_img = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    
    return final_img


def auto_correct_province(text: str) -> str:
    """
    Finds the closest matching Thai province name.
    """
    if not text:
        return ""
    
    # n=1 (get top 1 match), cutoff=0.3 (be very forgiving with OCR typos)
    matches = difflib.get_close_matches(text, VALID_PROVINCES, n=1, cutoff=0.3)
    
    return matches[0] if matches else text

def clean_plate_number(text: str) -> str:
    """
    Enforces Thai License Plate Grammar to fix OCR hallucinations.
    Format: [Optional Digit] + [1-2 Consonants] + [1-4 Digits]
    """
    text = text.replace(" ", "")
    
    # Common OCR mistakes where it reads a letter instead of a number
    num_confusions = {
        'ร': '5', 'ส': '5',
        'อ': '0', 'ด': '0', 'ถ': '0',
        'บ': '6', 
        'ง': '3', 
        'ท': '1', 'ก': '1', 'เ': '1', 'แ': '1', 'ไ': '1', 'ใ': '1', 'โ': '1', 'า': '1'
    }
    
    # Regex to capture the structure: (Prefix Number)(Consonants)(Suffix Numbers)
    match = re.match(r'^(\d?)([ก-ฮ]{0,2})(.*)$', text)
    
    
    if match:
        prefix_num = match.group(1)
        letters = match.group(2)
        suffix = match.group(3)
        print(f"DEBUG: Raw OCR Suffix before cleaning -> '{suffix}'")
        # Force the suffix to be strictly digits by fixing known confusions
        fixed_suffix = "".join([num_confusions.get(c, c) for c in suffix])
        # Strip out any remaining garbage characters that aren't digits
        fixed_suffix = re.sub(r'\D', '', fixed_suffix)
        
        return prefix_num + letters + fixed_suffix
        
    return text

def split_number_and_province(combined_text: str) -> Tuple[str, str]:
    digit_matches = list(re.finditer(r'\d', combined_text))
    if not digit_matches:
        return combined_text, ""
    last_digit_index = digit_matches[-1].end()
    return combined_text[:last_digit_index].strip(), combined_text[last_digit_index:].strip()

def extract_plate_text(image_path: str) -> Dict[str, str]:
    img = cv2.imread(image_path)
    if img is None:
        return {"number": "", "province": ""}
    clean_img = preprocess_image(img)
    detections = reader.readtext(clean_img, detail=1, allowlist=ALLOWLIST)
    raw_num_parts, pro_parts = [], []
    
    # Get the total height of the image
    image_h = clean_img.shape[0]
    
    # Create an invisible "Split Line" at 60% down the image height.
    # The number takes up most of the top space, province is usually at the very bottom.
    y_split_line = image_h * 0.60 

    for bbox, text, conf in detections:
        if conf < 0.10: 
            continue
            
        # bbox is a list of 4 points: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
        # bbox[0] is Top-Left [x, y], bbox[2] is Bottom-Right [x, y]
        xmin = bbox[0][0]
        ymin = bbox[0][1]
        ymax = bbox[2][1]
        
        # Calculate the vertical center of this specific text box
        y_center = (ymin + ymax) / 2
        
        # EXPERIMENT: SPATIAL SPLITTING
        # If the center of the text is ABOVE our split line, it's the Number.
        if y_center < y_split_line: 
            raw_num_parts.append((xmin, text))
        # If the center of the text is BELOW our split line, it's the Province.
        else: 
            pro_parts.append((xmin, text))

    combined_number_string = "".join([item[1] for item in sorted(raw_num_parts)]).strip()
    existing_province = "".join([item[1] for item in sorted(pro_parts)]).strip()

    # Still use the fallback split just in case EasyOCR merged them into one giant box
    extracted_num, extracted_pro = split_number_and_province(combined_number_string)
    
    # Apply Grammar Enforcement
    final_number = clean_plate_number(extracted_num)
    
    # Apply Auto-Correct
    raw_province = (extracted_pro + existing_province).strip()
    final_province = auto_correct_province(raw_province)

    return {"number": final_number, "province": final_province}

if __name__ == "__main__":
    # Quick Test Block
    test_path = "./test_images/3539f953-4c95-4393-84c4-2ec89d2e0fce_jpg.rf.4ca3bf773efcc26b44e6cf0973499903.jpg"
    if os.path.exists(test_path):
        result = extract_plate_text(test_path)
        print(f"Result: {result}")