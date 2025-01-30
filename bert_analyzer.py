import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from config.settings import CATEGORIES, MODEL_PATH

class TransactionAnalyzer:
    def __init__(self):
        self.categories = CATEGORIES
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'bert-base-uncased', 
            num_labels=len(self.categories)
        )
    
    def categorize_transactions(self, transactions_df):
        # Your existing categorize_transactions implementation
        pass
