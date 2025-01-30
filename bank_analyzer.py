import pytesseract
import pandas as pd
import pdfplumber
from config.settings import TESSERACT_PATH

class BankStatementAnalyzer:
    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        self.transactions_df = None
    
    def extract_from_pdf(self, pdf_path: str) -> pd.DataFrame:
        with pdfplumber.open(pdf_path) as pdf:
            all_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n"
            return self._parse_text(all_text)
    
    def _parse_text(self, text: str) -> pd.DataFrame:
        # Your existing _parse_text implementation
        pass
