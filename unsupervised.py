from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from config.settings import N_CLUSTERS

class UnsupervisedTransactionAnalyzer:
    def __init__(self, n_clusters=N_CLUSTERS):
        self.n_clusters = n_clusters
        self.tfidf = TfidfVectorizer(max_features=100)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.scaler = StandardScaler()
    
    def cluster_transactions(self, transactions_df):
        """Cluster transactions and return both DataFrame and analysis"""
        if transactions_df is None:
            raise ValueError("Input DataFrame is None")
            
        # Prepare features
        text_features, amount_features = self.prepare_features(transactions_df)
        
        # Combine features
        combined_features = np.hstack([
            text_features.toarray(),
            amount_features
        ])
        
        # Perform clustering
        clusters = self.kmeans.fit_predict(combined_features)
        
        # Add cluster labels to DataFrame
        transactions_df['Cluster'] = clusters
        
        # Analyze clusters
        cluster_analysis = {
            'cluster_sizes': pd.Series(clusters).value_counts().to_dict(),
            'cluster_means': transactions_df.groupby('Cluster')['Withdrawal'].mean().to_dict()
        }
        
        # Return both the DataFrame and analysis as a tuple
        return transactions_df, cluster_analysis
