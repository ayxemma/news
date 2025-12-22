"""
DataLoader class for loading all data sources
"""
import json
import pandas as pd
import numpy as np

class DataLoader:
    """Class for loading all data sources"""
    
    def load_loughran_mcdonald_complexity_words(self, filepath='data/Loughran-McDonald_MasterDictionary_1993-2024.csv'):
        """Load complexity words from Loughran-McDonald Master Dictionary CSV file."""
        print(f"Loading Loughran-McDonald complexity words from {filepath}...")
        try:
            df_dict = pd.read_csv(filepath)
            complexity_df = df_dict[df_dict['Complexity'] == 2009]
            complexity_words = set(complexity_df['Word'].str.lower().tolist())
            print(f"Loaded {len(complexity_words):,} complexity words from Loughran-McDonald dictionary")
            if len(complexity_words) > 0:
                print(f"Sample words: {list(complexity_words)[:10]}")
            return complexity_words
        except FileNotFoundError:
            print(f"Warning: File {filepath} not found. Using empty set.")
            return set()
        except Exception as e:
            print(f"Error loading dictionary file: {e}")
            print("Using empty set.")
            return set()
    
    def load_loughran_mcdonald_uncertainty_words(self, filepath='data/Loughran-McDonald_MasterDictionary_1993-2024.csv'):
        """Load uncertainty words from Loughran-McDonald Master Dictionary CSV file."""
        print(f"Loading Loughran-McDonald uncertainty words from {filepath}...")
        try:
            df_dict = pd.read_csv(filepath)
            uncertainty_df = df_dict[df_dict['Uncertainty'] == 2009]
            uncertainty_words = set(uncertainty_df['Word'].str.lower().tolist())
            print(f"Loaded {len(uncertainty_words):,} uncertainty words from Loughran-McDonald dictionary")
            print(f"Sample words: {list(uncertainty_words)[:10]}")
            return uncertainty_words
        except FileNotFoundError:
            print(f"Warning: File {filepath} not found. Using fallback list.")
            return set()
        except Exception as e:
            print(f"Error loading dictionary file: {e}")
            return set()
    
    def load_news_dataset(self, filepath='data/News_Category_Dataset_v3.json'):
        """Load news dataset from JSON file"""
        print("Loading news dataset...")
        news_data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    news_data.append(json.loads(line))
        df_news = pd.DataFrame(news_data)
        df_news['date'] = pd.to_datetime(df_news['date'])
        df_news = df_news.sort_values('date').reset_index(drop=True)
        print(f"Loaded {len(df_news):,} news articles")
        print(f"Date range: {df_news['date'].min()} to {df_news['date'].max()}")
        print(f"\nCategories: {df_news['category'].nunique()}")
        print(df_news['category'].value_counts().head(10))
        return df_news
    
    def load_spy_returns(self, filepath='data/SP500.csv'):
        """Load SPY returns data and calculate next-day returns"""
        df_spy = pd.read_csv(filepath)
        df_spy['Date'] = pd.to_datetime(df_spy['Date'])
        df_spy = df_spy.rename(columns={'Date': 'date'})
        df_spy = df_spy[df_spy['SP500'].notna()].copy()
        df_spy['spy_return'] = np.log(df_spy['SP500'] / df_spy['SP500'].shift(1))
        df_spy = df_spy[df_spy['spy_return'].notna()].copy()
        df_spy['spy_return_next'] = df_spy['spy_return'].shift(-1)
        print(f"SPY data loaded: {len(df_spy):,} trading days")
        print(f"Date range: {df_spy['date'].min()} to {df_spy['date'].max()}")
        print(f"\nReturn statistics:")
        print(df_spy['spy_return'].describe())
        return df_spy

