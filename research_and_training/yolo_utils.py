def clip_value(value, min_val=0.0, max_val=1.0):
    return max(min_val, min(value, max_val))

def convert_to_yolo(bbox, img_width, img_height, is_normalized=False, bbox_format="pascal"):
    if is_normalized:
        cx, cy, w, h = bbox
        return [clip_value(cx), clip_value(cy), clip_value(w), clip_value(h)]
    
    if bbox_format == "coco":
        x_min, y_min, w_abs, h_abs = bbox
        xmin = max(0, x_min)
        ymin = max(0, y_min)
        xmax = min(img_width, x_min + w_abs)
        ymax = min(img_height, y_min + h_abs)
        
        w_abs = xmax - xmin
        h_abs = ymax - ymin
        cx_abs = xmin + (w_abs / 2.0)
        cy_abs = ymin + (h_abs / 2.0)
        
        cx = cx_abs / img_width
        cy = cy_abs / img_height
        w = w_abs / img_width
        h = h_abs / img_height
        return [clip_value(cx), clip_value(cy), clip_value(w), clip_value(h)]
    
    xmin, ymin, xmax, ymax = bbox
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_width, xmax)
    ymax = min(img_height, ymax)

    w_abs = xmax - xmin
    h_abs = ymax - ymin
    cx_abs = xmin + (w_abs / 2.0)
    cy_abs = ymin + (h_abs / 2.0)

    cx = cx_abs / img_width
    cy = cy_abs / img_height
    w = w_abs / img_width
    h = h_abs / img_height

    return [clip_value(cx), clip_value(cy), clip_value(w), clip_value(h)]
