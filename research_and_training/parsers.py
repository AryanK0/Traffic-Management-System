import xml.etree.ElementTree as ET
import json

def parse_xml(file_path):
    boxes = []
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            label = obj.find('name').text
            bndbox = obj.find('bndbox')
            if bndbox is not None:
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                boxes.append({"label": label, "bbox": [xmin, ymin, xmax, ymax]})
    except Exception:
        pass
    return boxes

_coco_cache = {}

def parse_coco_monolithic(file_path, img_basename):
    if file_path not in _coco_cache:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        img_id_to_name = {img['id']: img['file_name'] for img in data.get('images', [])}
        cat_id_to_name = {cat['id']: cat['name'] for cat in data.get('categories', [])}
        
        name_to_boxes = {}
        for ann in data.get('annotations', []):
            img_name_raw = img_id_to_name.get(ann['image_id'])
            if not img_name_raw:
                continue
            
            # Map the exact basename rather than relative directory paths like 000/100103.png
            import os
            img_name = os.path.basename(img_name_raw)
            
            cat_name = cat_id_to_name.get(ann['category_id'], 'unknown')
            if img_name not in name_to_boxes:
                name_to_boxes[img_name] = []
            
            bbox = ann.get('bbox', [])
            if len(bbox) == 4:
                name_to_boxes[img_name].append({
                    "label": cat_name, 
                    "bbox": bbox, 
                    "format": "coco"
                })
        
        _coco_cache[file_path] = name_to_boxes
        
    return _coco_cache[file_path].get(img_basename, [])

def parse_json(file_path, img_basename=None):
    if img_basename and 'UVH' in file_path:
        return parse_coco_monolithic(file_path, img_basename)

    boxes = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            annotations = data.get('annotations', [])
            if not annotations and 'shapes' in data:
                annotations = data['shapes']
            for ann in annotations:
                if 'bbox' in ann and 'label' in ann:
                    bbox = ann['bbox']
                    boxes.append({"label": ann['label'], "bbox": bbox})
                elif 'points' in ann and 'label' in ann:
                    pts = ann['points']
                    xmin = min([p[0] for p in pts])
                    ymin = min([p[1] for p in pts])
                    xmax = max([p[0] for p in pts])
                    ymax = max([p[1] for p in pts])
                    boxes.append({"label": ann['label'], "bbox": [xmin, ymin, xmax, ymax]})
        elif isinstance(data, list):
            for ann in data:
                if 'bbox' in ann and 'label' in ann:
                    boxes.append({"label": ann['label'], "bbox": ann['bbox']})
    except Exception:
        pass
    return boxes

def parse_txt(file_path):
    boxes = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                label = parts[0]
                bbox = [float(x) for x in parts[1:5]]
                boxes.append({"label": label, "bbox": bbox})
    except Exception:
        pass
    return boxes
