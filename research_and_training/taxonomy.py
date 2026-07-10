import os
import csv
from config import CLASS_MAPPING, LOG_FILE

def init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['image_path', 'raw_label'])

def log_unknown(image_path, raw_label):
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([image_path, raw_label])

def get_class_id(raw_label, image_path=None):
    raw_lower = str(raw_label).lower()
    for class_id, keywords in CLASS_MAPPING.items():
        for kw in keywords:
            if kw in raw_lower:
                return class_id
    
    if image_path:
        log_unknown(image_path, raw_label)
    return 4
