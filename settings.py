import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_PATH = os.path.join(BASE_DIR, '..', 'data', 'input')
OUTPUT_PATH = os.path.join(BASE_DIR, '..', 'data', 'output')
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'bert_model')
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Model settings
CATEGORIES = [
    'Food & Dining', 'Shopping', 'Rent', 'Transfer', 
    'Utilities', 'Entertainment', 'Transportation'
]
N_CLUSTERS = 5
