"""
Feature extraction functions for news articles
"""
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from textstat import flesch_reading_ease, gunning_fog, dale_chall_readability_score
from tqdm import tqdm
import os
import hashlib
from data_loader import DataLoader

import os
os.chdir('/Users/ayx/Documents/Projects/news')
print(os.getcwd())

# Check if pyarrow is available
try:
    import pyarrow
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False


class ArticleFeatureExtractor:
    """Class for extracting features from news articles"""
    
    def initialize_sentiment_model(self):
        """Initialize sentiment analysis model"""
        print("Loading sentiment model...")
        
        # Set torch to use single thread to avoid deadlocks in Jupyter
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        try:
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                use_safetensors=True,
                torch_dtype=torch.float32
            )
        except:
            print("Warning: safetensors not available, trying standard loading (requires torch >= 2.6)")
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            )
        
        # Force CPU mode to avoid CUDA issues
        self.sentiment_device = torch.device("cpu")
        print(f"Using device: {self.sentiment_device}")
        self.sentiment_model.to(self.sentiment_device)
        self.sentiment_model.eval()
        
        # Disable gradient computation globally for this model
        for param in self.sentiment_model.parameters():
            param.requires_grad = False
        
        print(f"Model loaded on device: {self.sentiment_device}")
        
        self.sentiment_labels = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    def get_sentiment(self, text):
        """Get sentiment scores for a single text"""
        if not text or not isinstance(text, str):
            return {
                'label': 'neutral',
                'score': 0,
                'negative': 0,
                'neutral': 1,
                'positive': 0
            }
        
        inputs = self.sentiment_tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(self.sentiment_device) for k, v in inputs.items()}
        
        # Use inference_mode for better performance and stability (fallback to no_grad for older PyTorch)
        try:
            inference_context = torch.inference_mode()
        except AttributeError:
            inference_context = torch.no_grad()
        
        with inference_context:
            try:
                outputs = self.sentiment_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            except Exception as e:
                print(f"    Error during model inference: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        prob = probs[0]
        label_idx = prob.argmax().item()
        
        # Calculate sentiment score as positive_prob - negative_prob
        # This gives a continuous score from -1 (very negative) to +1 (very positive)
        sentiment_score = prob[2].item() - prob[0].item()
        
        return {
            'label': self.sentiment_labels[label_idx],
            'score': sentiment_score,
            'negative': prob[0].item(),
            'neutral': prob[1].item(),
            'positive': prob[2].item()
        }
    
    def __init__(self, data_loader=None):
        """Initialize the feature extractor
        
        Args:
            data_loader: Optional DataLoader instance (creates new one if not provided)
        """
        if data_loader is None:
            data_loader = DataLoader()
        self.data_loader = data_loader
        self.uncertainty_words = self.data_loader.load_loughran_mcdonald_uncertainty_words()
        
        # Initialize sentiment model
        self.initialize_sentiment_model()
    
    def compute_basic_features(self, df):
        """Compute basic features: token count and length"""
        df = df.copy()
        df['headline_tokens'] = df['headline'].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
        df['headline_length'] = df['headline'].apply(
            lambda x: len(str(x)) if pd.notna(x) else 0
        )
        return df
    
    def compute_complexity_features(self, df):
        """Compute complexity features: readability metrics"""
        print("Computing complexity features...")
        df = df.copy()
        
        def compute_complexity(text):
            if not text or not isinstance(text, str):
                return {'flesch': 0, 'gunning_fog': 0, 'dale_chall': 0}
            try:
                # Readability metrics
                flesch_score = flesch_reading_ease(text)
                gunning_fog_score = gunning_fog(text)
                dale_chall_score = dale_chall_readability_score(text)
                
                return {
                    'flesch': flesch_score,
                    'gunning_fog': gunning_fog_score,
                    'dale_chall': dale_chall_score
                }
            except:
                return {'flesch': 0, 'gunning_fog': 0, 'dale_chall': 0}
        
        complexity = df['headline'].apply(compute_complexity)
        df['flesch'] = complexity.apply(lambda x: x['flesch'])
        df['gunning_fog'] = complexity.apply(lambda x: x['gunning_fog'])
        df['dale_chall'] = complexity.apply(lambda x: x['dale_chall'])

        df = self.normalize_complexity_features(df, 'flesch')
        df = self.normalize_complexity_features(df, 'gunning_fog')
        df = self.normalize_complexity_features(df, 'dale_chall')
        df['complexity'] = (df['flesch_normalized'] + df['gunning_fog_normalized'] + df['dale_chall_normalized']) / 3
        return df

    def normalize_complexity_features(self, df, feature):
        window_size = 90
        min_periods = 30
        df[f'{feature}_normalized'] = df[feature].rolling(window=window_size, min_periods=min_periods).mean() / df[feature].rolling(window=window_size, min_periods=min_periods).std()
        return df
    
    def compute_uncertainty_features(self, df):
        """Compute uncertainty features: uncertainty word count and ratio"""
        print("Computing uncertainty features...")
        df = df.copy()
        
        def count_uncertainty(text):
            if not text or not isinstance(text, str):
                return {'uncertainty_count': 0, 'uncertainty_ratio': 0}
            words = text.lower().split()
            total_words = len(words)
            uncertainty_count = sum(1 for word in words if word in self.uncertainty_words)
            return {
                'uncertainty_count': uncertainty_count,
                'uncertainty_ratio': uncertainty_count / total_words if total_words > 0 else 0
            }
        
        uncertainty = df['headline'].apply(count_uncertainty)
        df['uncertainty_count'] = uncertainty.apply(lambda x: x['uncertainty_count'])
        df['uncertainty_ratio'] = uncertainty.apply(lambda x: x['uncertainty_ratio'])
        
        return df
    
    def save_sentiment_results(self, df, filepath='data/cache/sentiment_cache.parquet'):
        """Save sentiment results to parquet file (overwrites existing file)
        
        Args:
            df: DataFrame with sentiment columns (must include 'headline' column)
            filepath: Path to save the parquet file
        """
        if not PYARROW_AVAILABLE:
            error_msg = (
                "\n" + "="*60 + "\n"
                "ERROR: pyarrow is required for parquet support.\n"
                "Please install it using one of the following:\n"
                "  - pip install pyarrow\n"
                "  - conda install pyarrow\n"
                "  - conda install -c conda-forge pyarrow\n"
                "\n"
                "After installing, restart your kernel and try again.\n"
                "="*60
            )
            raise ImportError(error_msg)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Save directly without merging with existing cache
        df.to_parquet(filepath, index=False, engine='pyarrow')
        print(f"Sentiment results saved to {filepath} ({len(df):,} records)")
    
    def load_sentiment_results(self, filepath='data/cache/sentiment_cache.parquet'):
        """Load sentiment results from parquet file
        
        Args:
            filepath: Path to the parquet file
            
        Returns:
            DataFrame with sentiment results, or None if file doesn't exist
        """
        if not PYARROW_AVAILABLE:
            error_msg = (
                "\n" + "="*60 + "\n"
                "ERROR: pyarrow is required for parquet support.\n"
                "Please install it using one of the following:\n"
                "  - pip install pyarrow\n"
                "  - conda install pyarrow\n"
                "  - conda install -c conda-forge pyarrow\n"
                "\n"
                "After installing, restart your kernel and try again.\n"
                "="*60
            )
            print(error_msg)
            return None
        
        if not os.path.exists(filepath):
            return None
        
        try:
            df = pd.read_parquet(filepath)
            print(f"Loaded {len(df):,} sentiment results from {filepath}")
            return df
        except Exception as e:
            print(f"Error loading sentiment cache: {e}")
            return None
    
    def add_sentiment_features(self, df, reload=False):
        if reload:
            sentiment_df = self.load_sentiment_results()
        else:
            sentiment_df = self.compute_sentiment_features(df)
            self.save_sentiment_results(sentiment_df)
        for col in ['sentiment_label', 'sentiment_score', 'sentiment_negative', 
                   'sentiment_neutral', 'sentiment_positive']:
            df[col] = sentiment_df[col].values
        return df
    
    def compute_sentiment_features(self, df):
        """Compute sentiment features using the sentiment function
        
        Args:
            df: DataFrame with headlines
            reload: If True, load from cache file and return. If False, always recalculate and save.
            cache_file: Path to cache parquet file
        """
        df = df.copy()
        
        # Compute sentiment for all headlines
        print("Computing sentiment features (this may take a while)...")
        print(f"Total articles to process: {len(df):,}")
        
        # Check if DataFrame is empty
        if len(df) == 0:
            print("Warning: DataFrame is empty!")
            return df
        
        sentiment_results = []
        total_rows = len(df)
        
        print("Starting sentiment processing loop...")
        
        # Simple for loop through rows
        for i in tqdm(range(total_rows)):
            
            headline = df.iloc[i]['headline']
            if not headline or pd.isna(headline):
                sentiment_results.append({
                    'sentiment_label': 'neutral',
                    'sentiment_score': 0,
                    'sentiment_negative': 0,
                    'sentiment_neutral': 1,
                    'sentiment_positive': 0
                })
                continue
            
            sent = self.get_sentiment(headline)
            
            sentiment_results.append({
                'sentiment_label': sent['label'],
                'sentiment_score': sent['score'],
                'sentiment_negative': sent['negative'],
                'sentiment_neutral': sent['neutral'],
                'sentiment_positive': sent['positive']
            })
        
        print(f"Completed processing {len(sentiment_results)} articles")
        
        # Add computed sentiment results
        sentiment_df = pd.DataFrame(sentiment_results)
        
        return sentiment_df
  
    
    def compute_all_features(self, df_news, reload_cache=False):
        """Compute all features for news articles"""
        print("Computing features on individual articles...")
        
        df_features = df_news.copy()  
        # Compute features in sequence
        df_features = self.compute_basic_features(df_features)
        df_features = self.compute_complexity_features(df_features)
        df_features = self.compute_uncertainty_features(df_features)
        df_features = self.add_sentiment_features(df_features, reload=reload_cache)
        
        print(f"Features computed for {len(df_features):,} articles")
        return df_features


def run():
    from data_loader import DataLoader
    data_loader = DataLoader()
    df_news = data_loader.load_news_dataset()
    extractor = ArticleFeatureExtractor(data_loader=data_loader)
    # df_news = df_news.head(100).copy()
    df_features = extractor.compute_all_features(df_news, reload_cache=True)
    print(df_features.head())



