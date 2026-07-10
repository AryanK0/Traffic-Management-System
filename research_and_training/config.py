import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIRS = {
    'idd': os.path.join(BASE_DIR, 'dataset', 'IDD117K_Detection'),
    'fgvd': os.path.join(BASE_DIR, 'dataset', 'IDD_FGVD'),
    'i2wdd': os.path.join(BASE_DIR, 'dataset', 'i2wdd'),
    'dats': os.path.join(BASE_DIR, 'dataset', 'dats'),
    'uvh26': os.path.join(BASE_DIR, 'dataset', 'UVH-26'),
    'iruvd': os.path.join(BASE_DIR, 'dataset', 'IRUVD')
}

OUTPUT_DIR = os.path.join(BASE_DIR, 'master_dataset')
OUTPUT_IMAGES_TRAIN = os.path.join(OUTPUT_DIR, 'images', 'train')
OUTPUT_IMAGES_VAL = os.path.join(OUTPUT_DIR, 'images', 'val')
OUTPUT_LABELS_TRAIN = os.path.join(OUTPUT_DIR, 'labels', 'train')
OUTPUT_LABELS_VAL = os.path.join(OUTPUT_DIR, 'labels', 'val')

LOG_FILE = os.path.join(BASE_DIR, 'unknown_classes_log.csv')

CLASS_MAPPING = {
    0: ['2-wheeler', 'bike', 'motor-rickshaw', 'cycle', 'motorcycle', 'scooter', '2_wheeler', 'two_wheeler', 'motor'],
    1: ['3-wheeler', 'auto-rickshaw', 'toto', 'cycle-rickshaw', 'autorickshaw', 'auto', 'e-rickshaw', 'rickshaw', '3_wheeler', 'three_wheeler', 'tuk_tuk', 'tuktuk'],
    2: ['hatchback', 'sedan', 'suv', 'muv', 'van', 'car', 'jeep', 'taxi', '4_wheeler', 'four_wheeler', 'marutisuzuki'],
    3: ['lcv', 'mini-bus', 'bus', 'truck', 'tempo-traveller', 'tempo', 'tractor', 'lorry', 'heavy_vehicle', 'trailer']
}
