import glob
import os
import shutil

import pandas as pd

CHAR_DECODER = {
    "TH01": "ก", "TH02": "ข", "TH03": "ค", "TH04": "ฆ", "TH05": "ง",
    "TH06": "จ", "TH07": "ฉ", "TH08": "ช", "TH09": "ฌ", "TH10": "ญ",
    "TH11": "ฎ", "TH12": "ฐ", "TH13": "ฒ", "TH14": "ณ", "TH15": "ด",
    "TH16": "ต", "TH17": "ถ", "TH18": "ท", "TH19": "ธ", "TH20": "น",
    "TH21": "บ", "TH22": "ผ", "TH23": "พ", "TH24": "ฟ", "TH25": "ภ",
    "TH26": "ม", "TH27": "ย", "TH28": "ร", "TH29": "ล", "TH30": "ว",
    "TH31": "ศ", "TH32": "ษ", "TH33": "ส", "TH34": "ห", "TH35": "ฬ",
    "TH36": "อ", "TH37": "ฮ"
}

PROVINCE_MAP = {
    "ATG": "อ่างทอง", "AYA": "พระนครศรีอยุธยา", "BKK": "กรุงเทพมหานคร",
    "BKN": "บึงกาฬ", "BRM": "บุรีรัมย์", "CBI": "ชลบุรี",
    "CCO": "ฉะเชิงเทรา", "CMI": "เชียงใหม่", "CNT": "ชัยนาท",
    "CPM": "ชัยภูมิ", "CPN": "ชุมพร", "CRI": "เชียงราย",
    "CTI": "จันทบุรี", "KBI": "กระบี่", "KKN": "ขอนแก่น",
    "KPT": "กำแพงเพชร", "KRI": "กาญจนบุรี", "KSN": "กาฬสินธุ์",
    "LEI": "เลย", "LPG": "ลำปาง", "LPN": "ลำพูน",
    "LRI": "ลพบุรี", "MDH": "มุกดาหาร", "MKM": "มหาสารคาม",
    "NAN": "น่าน", "NBI": "นนทบุรี", "NBP": "หนองบัวลำภู",
    "NKI": "หนองคาย", "NMA": "นครราชสีมา", "NPM": "นครพนม",
    "NPT": "นครปฐม", "NSN": "นครสวรรค์", "NST": "นครศรีธรรมราช",
    "NYK": "นครนายก", "PBI": "เพชรบุรี", "PCT": "พิจิตร",
    "PKN": "ประจวบคีรีขันธ์", "PKT": "ภูเก็ต", "PLG": "พัทลุง",
    "PLK": "พิษณุโลก", "PNB": "พังงา", "PRE": "แพร่",
    "PRI": "ปราจีนบุรี", "PTE": "ปทุมธานี", "PYO": "พะเยา",
    "RBR": "ราชบุรี", "RET": "ร้อยเอ็ด", "RYG": "ระยอง",
    "SBR": "สระบุรี", "SKA": "สงขลา", "SKM": "สมุทรสงคราม",
    "SKN": "สมุทรสาคร", "SKW": "สระแก้ว", "SNI": "สุราษฎร์ธานี",
    "SNK": "สกลนคร", "SPB": "สุพรรณบุรี", "SPK": "สมุทรปราการ",
    "SRI": "สุรินทร์", "SRN": "สิงห์บุรี", "SSK": "ศรีสะเกษ",
    "STI": "สุโขทัย", "TAK": "ตาก", "TRT": "ตราด"
}

def main():
    target_dir = "./test_images/"
    # Clean out the old images if they exist so you don't mix train/test
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    
    master_labels = []

    # CHANGE: Target the 'test' folder specifically
    search_path = "dataset/LPR plate.v3-new.tensorflow/test/*.csv"
    csv_files = glob.glob(search_path, recursive=True)
    
    if not csv_files:
        print(f"Warning: No CSV files found in {search_path}")
        return

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        img_folder = os.path.dirname(csv_path)

        for filename, group in df.groupby("filename"):
            # Sort boxes left-to-right to build the plate number correctly
            sorted_rows = group.sort_values(by="xmin")
            plate_chars = []
            province = ""

            for _, row in sorted_rows.iterrows():
                cls = str(row["class"]).strip()
                if cls in PROVINCE_MAP:
                    province = PROVINCE_MAP[cls]
                elif cls in CHAR_DECODER:
                    plate_chars.append(CHAR_DECODER[cls])
                elif cls.isdigit():
                    plate_chars.append(cls)

            true_num = "".join(plate_chars)
            src = os.path.join(img_folder, filename)
            dst = os.path.join(target_dir, filename)
            
            if os.path.exists(src):
                shutil.copy2(src, dst)
                master_labels.append({
                    "image_filename": filename, 
                    "true_number": true_num, 
                    "true_province": province
                })

    # Save to a new labels file for your evaluation script
    pd.DataFrame(master_labels).to_csv("labels.csv", index=False, encoding="utf-8-sig")
    print(f"Test Labels built! Total: {len(master_labels)}")

if __name__ == "__main__":
    main()
