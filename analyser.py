



import pytesseract
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import pdfplumber
import io
from PIL import Image
import numpy as np

class BankStatementAnalyzer:
    def __init__(self):
        # Configure Tesseract path for Windows
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        self.transactions_df = None
    
    def extract_from_pdf(self, pdf_path: str) -> pd.DataFrame:
        """Extract text from PDF using pdfplumber"""
        with pdfplumber.open(pdf_path) as pdf:
            all_text = ""
            for page in pdf.pages:
                # Extract text from each page
                text = page.extract_text()
                if text:
                    all_text += text + "\n"
            
            return self._parse_text(all_text)
    
    def _parse_text(self, text: str) -> pd.DataFrame:
        """Parse extracted text into structured data"""
        transactions = []
        lines = text.split('\n')
        
        for line in lines:
            # Skip empty lines and headers
            if not line.strip() or 'Date' in line or 'Opening Balance' in line:
                continue
                
            parts = line.split()
            if len(parts) >= 3:
                try:
                    # Extract date
                    date = parts[0]
                    
                    # Extract reference number
                    ref_no = next((part for part in parts if part.startswith('UPI-')), '')
                    
                    # Extract amounts
                    withdrawal = 0.0
                    deposit = 0.0
                    balance = 0.0
                    
                    # Process amounts
                    for i, part in enumerate(reversed(parts)):
                        try:
                            amount = float(part.replace(',', '').replace('(Cr)', ''))
                            if '(Cr)' in part:
                                if i == 0:  # Last number is balance
                                    balance = amount
                                else:
                                    deposit = amount
                            else:
                                withdrawal = amount
                        except ValueError:
                            continue
                    
                    # Get description
                    desc_parts = [p for p in parts[1:] if not p.startswith('UPI-') and 
                                not any(c.isdigit() for c in p.replace(',', ''))]
                    description = ' '.join(desc_parts)
                    
                    transactions.append({
                        'Date': date,
                        'Description': description,
                        'Reference': ref_no,
                        'Withdrawal': withdrawal,
                        'Deposit': deposit,
                        'Balance': balance
                    })
                except Exception as e:
                    continue
        
        self.transactions_df = pd.DataFrame(transactions)
        return self.transactions_df

    def analyze_transactions(self) -> dict:
        if self.transactions_df is None:
            raise ValueError("No transaction data available")
        
        return {
            'total_transactions': len(self.transactions_df),
            'total_deposits': self.transactions_df['Deposit'].sum(),
            'total_withdrawals': self.transactions_df['Withdrawal'].sum(),
            'opening_balance': self.transactions_df['Balance'].iloc[0],
            'closing_balance': self.transactions_df['Balance'].iloc[-1]
        }


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class TransactionAnalyzer:
    def __init__(self):
        # Define transaction categories
        self.categories = [
            'Food & Dining', 'Shopping', 'Rent', 'Transfer', 
            'Utilities', 'Entertainment', 'Transportation'
        ]
        
        # Load BERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'bert-base-uncased', 
            num_labels=len(self.categories)
        )
    
    def categorize_transactions(self, transactions_df):
        """Categorize transactions using BERT"""
        descriptions = transactions_df['Narration'].str.lower().tolist()
        
        # Tokenize descriptions
        inputs = self.tokenizer(
            descriptions,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            categories = [self.categories[pred] for pred in predictions.argmax(dim=1)]
        
        # Add categories to dataframe
        transactions_df['Category'] = categories
        return transactions_df

def create_visualizations(transactions_df):
    """Create visualizations for the transactions"""
    plt.style.use('default')
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Pie Chart: Expenses by Category
    plt.subplot(2, 2, 1)
    category_expenses = transactions_df.groupby('Category')['Withdrawal'].sum()
    plt.pie(category_expenses, labels=category_expenses.index, autopct='%1.1f%%', 
            colors=plt.cm.Set3(np.linspace(0, 1, len(category_expenses))))
    plt.title('Expenses Distribution by Category', pad=20)
    
    # 2. Bar Chart: Category-wise Spending
    plt.subplot(2, 2, 2)
    category_expenses.plot(kind='bar')
    plt.title('Category-wise Spending')
    plt.ylabel('Amount (₹)')
    plt.xticks(rotation=45)
    
    # 3. Balance Trend
    plt.subplot(2, 2, 3)
    transactions_df['Date'] = pd.to_datetime(transactions_df['Date'], format='%d-%b-%y')
    plt.plot(transactions_df['Date'], transactions_df['Balance'], 
             marker='o', linewidth=2, markersize=8)
    plt.title('Balance Trend')
    plt.xticks(rotation=45)
    plt.ylabel('Balance (₹)')
    
    # 4. Daily Transactions
    plt.subplot(2, 2, 4)
    daily_trans = transactions_df.groupby('Date')[['Withdrawal', 'Deposit']].sum()
    daily_trans.plot(kind='bar')
    plt.title('Daily Transaction Activity')
    plt.xticks(rotation=45)
    plt.ylabel('Amount (₹)')
    
    plt.tight_layout()
    plt.savefig('transaction_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Create sample transaction data
    data = {
        'Date': ['01-Apr-23', '01-Apr-23', '03-Apr-23', '03-Apr-23', '05-Apr-23', '05-Apr-23', 
                 '05-Apr-23', '06-Apr-23', '07-Apr-23', '08-Apr-23', '08-Apr-23', '08-Apr-23', 
                 '08-Apr-23', '08-Apr-23', '10-Apr-23'],
        'Narration': ['OPENING BALANCE', 'UPI/PRINCE GAKHAR', 'UPI/SUNIL MANI', 'UPI/Judith Lamhoich',
                      'UPI/Swiggy', 'UPI/PRINCE GAKHAR', 'UPI/Dough N Cream', 'UPI/DASHRATH KUMAR',
                      'UPI/Honasa Consumer', 'UPI/PRASHANT TIWARI', 'UPI/SUNIL MANI', 'UPI/Judith Lamhoich',
                      'UPI/Fruits Shop', 'UPI/Anuj Dairy', 'UPI/Santosh'],
        'Withdrawal': [0, 27.00, 0, 3000.00, 346.00, 40.00, 567.00, 121.00, 505.92, 0, 0, 6400.00,
                       140.00, 100.00, 20.00],
        'Deposit': [0, 0, 5000.00, 0, 0, 0, 0, 0, 0, 700.00, 6400.00, 0, 0, 0, 0],
        'Balance': [10105.68, 10078.68, 15078.68, 12078.68, 11732.68, 11692.68, 11125.68, 11004.68,
                    10497.76, 11197.76, 17597.76, 11197.76, 11057.76, 10957.76, 10937.76]
    }

    # Create DataFrame
    df = pd.DataFrame(data)
    
    try:
        # First categorize transactions
        analyzer = TransactionAnalyzer()
        categorized_df = analyzer.categorize_transactions(df)
        
    
        # Then create visualizations
        create_visualizations(categorized_df)
        
        # Print analysis
        print("\nTransaction Categories:")
        print(categorized_df.groupby('Category')['Withdrawal'].sum())
        
    except Exception as e:
        print(f"Error: {str(e)}")
