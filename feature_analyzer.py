"""
Feature aggregation, alignment, and analysis functions
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns


class FeatureAnalyzer:
    """Class for aggregating features, aligning with returns, and analyzing feature performance"""
    
    def _build_aggregation_dict(self, df_features):
        """Build aggregation dictionary for required features"""
        agg_dict = {}
        if 'sentiment_score' in df_features.columns:
            agg_dict['sentiment_score'] = 'mean'
        if 'complexity' in df_features.columns:
            agg_dict['complexity'] = 'mean'
        if 'headline_tokens' in df_features.columns:
            agg_dict['headline_tokens'] = 'mean'
        return agg_dict
    
    def _compute_sentiment_ratio(self, df_features, groupby_keys):
        """Compute sentiment ratio: (positive_count - negative_count) / total_count"""
        if 'sentiment_label' not in df_features.columns:
            return None
        
        def count_sentiments(group):
            total = len(group)
            positive_count = sum(group == 'positive')
            negative_count = sum(group == 'negative')
            return (positive_count - negative_count) / total if total > 0 else 0
        
        sentiment_ratios = df_features.groupby(groupby_keys)['sentiment_label'].apply(count_sentiments).reset_index()
        sentiment_ratios.columns = list(groupby_keys) + ['sentiment_ratio']
        return sentiment_ratios
    
    def normalize_features_rolling_zscore(self, df, feature_cols=None, date_col='date', 
                                         window_size=90, min_periods=30):
        """Normalize features using rolling window time-series z-score
        
        Applies rolling window z-score normalization to specified features. The dataframe should
        be indexed by date (daily frequency) with each column as a feature. This normalization
        helps remove trends and makes features more comparable across time periods while avoiding
        look-ahead bias by only using historical data.
        
        Args:
            df: DataFrame with features to normalize (must be sorted by date, daily frequency)
            feature_cols: List of feature column names to normalize. If None, uses default features:
                         ['sentiment_score', 'complexity', 'sentiment_ratio', 'headline_count', 'average_token_length']
            date_col: Name of date column for sorting (default: 'date')
            window_size: Rolling window size in days (default: 90)
            min_periods: Minimum periods required for rolling calculation (default: 30)
            
        Returns:
            DataFrame with normalized features (original columns + '_normalized' columns)
            
        Example:
            >>> analyzer = FeatureAnalyzer()
            >>> daily_features = analyzer.aggregate_features_to_daily(df_features)
            >>> normalized = analyzer.normalize_features_rolling_zscore(daily_features)
        """
        df = df.copy()
        
        # Default feature columns if not specified
        if feature_cols is None:
            feature_cols = ['sentiment_score', 'complexity', 'sentiment_ratio', 
                          'headline_count', 'average_token_length']
        
        # Ensure dates are sorted
        if date_col in df.columns:
            df = df.sort_values(date_col).reset_index(drop=True)
        
        # Features to normalize (only those that exist)
        features_to_normalize = [col for col in feature_cols if col in df.columns]
        
        if not features_to_normalize:
            print("Warning: No features found to normalize")
            return df
        
        print(f"Normalizing {len(features_to_normalize)} features using rolling window z-score...")
        print(f"Window size: {window_size} days, Min periods: {min_periods}")
        
        # Apply rolling window z-score normalization to each feature
        for feature in features_to_normalize:
            rolling_mean = df[feature].rolling(window=window_size, min_periods=min_periods).mean()
            rolling_std = df[feature].rolling(window=window_size, min_periods=min_periods).std()
            
            # Z-score normalization: (value - rolling_mean) / rolling_std
            df[f'{feature}_normalized'] = np.where(
                rolling_std > 0,
                (df[feature] - rolling_mean) / rolling_std,
                np.nan
            )
        
        # Forward fill NaN values (for early periods with insufficient history)
        normalized_cols = [f'{f}_normalized' for f in features_to_normalize]
        df[normalized_cols] = df[normalized_cols].ffill().fillna(0)
        
        print(f"Created normalized columns: {', '.join(normalized_cols)}")
        
        return df
    
    def _aggregate_features_core(self, df_features, groupby_keys, print_prefix=""):
        """Core aggregation logic shared by both daily and category aggregations
        
        Args:
            df_features: DataFrame with article-level features
            groupby_keys: List of column names to group by (e.g., ['date'] or ['date', 'category'])
            print_prefix: Optional prefix for print statements
            
        Returns:
            DataFrame with aggregated features
        """
        # Build aggregation dictionary
        agg_dict = self._build_aggregation_dict(df_features)
        
        # Aggregate numeric features
        if agg_dict:
            aggregated = df_features.groupby(groupby_keys).agg(agg_dict).reset_index()
            # Rename headline_tokens to average_token_length
            if 'headline_tokens' in aggregated.columns:
                aggregated = aggregated.rename(columns={'headline_tokens': 'average_token_length'})
        else:
            # If no features to aggregate, create empty dataframe with groupby keys
            aggregated = df_features[groupby_keys].drop_duplicates()
        
        # Add headline count
        headline_counts = df_features.groupby(groupby_keys).size().reset_index(name='headline_count')
        aggregated = aggregated.merge(headline_counts, on=groupby_keys, how='left')
        aggregated['headline_count'] = aggregated['headline_count'].fillna(0)
        
        # Compute sentiment ratio
        sentiment_ratios = self._compute_sentiment_ratio(df_features, groupby_keys)
        if sentiment_ratios is not None:
            aggregated = aggregated.merge(sentiment_ratios, on=groupby_keys, how='left')
            aggregated['sentiment_ratio'] = aggregated['sentiment_ratio'].fillna(0)
        
        # Sort by groupby keys
        aggregated = aggregated.sort_values(groupby_keys).reset_index(drop=True)
        
        return aggregated
    
    def aggregate_features_to_daily(self, df_features):
        """Aggregate article-level features to daily frequency
        
        Aggregates only:
        - sentiment_score (mean)
        - complexity (mean)
        - sentiment_ratio (positive_count - negative_count) / total_count
        - headline_count (total number of news articles)
        - average_token_length (mean of headline_tokens)
        """
        print("Aggregating features to daily frequency...")
        
        daily_features = self._aggregate_features_core(df_features, ['date'])
        
        print(f"Aggregated to {len(daily_features):,} unique dates")
        print(f"Average headlines per day: {daily_features['headline_count'].mean():.1f}")
        
        return daily_features
    
    def aggregate_features_by_category_to_daily(self, df_features):
        """Aggregate article-level features to daily frequency by category
        
        Aggregates only:
        - sentiment_score (mean)
        - complexity (mean)
        - sentiment_ratio (positive_count - negative_count) / total_count
        - headline_count (total number of news articles)
        - average_token_length (mean of headline_tokens)
        
        Args:
            df_features: DataFrame with article-level features including 'date' and 'category' columns
            
        Returns:
            DataFrame with daily features aggregated by category
        """
        print("Aggregating features to daily frequency by category...")
        
        # Ensure required columns exist
        required_cols = ['date', 'category']
        missing_cols = [col for col in required_cols if col not in df_features.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        daily_features_by_category = self._aggregate_features_core(df_features, ['date', 'category'])
        
        print(f"Aggregated to {len(daily_features_by_category):,} date-category combinations")
        print(f"Unique categories: {daily_features_by_category['category'].nunique()}")
        print(f"Date range: {daily_features_by_category['date'].min()} to {daily_features_by_category['date'].max()}")
        
        return daily_features_by_category
    
    def aggregate_and_normalize_features(self, df_features, window_size=90, min_periods=30):
        """Aggregate features to daily, merge with category-aggregated features, and normalize
        
        This function:
        1. Aggregates features to daily frequency (overall)
        2. Aggregates features to daily frequency by category
        3. Renames category feature columns to category_feature_name format
        4. Merges both dataframes on date
        5. Applies rolling window z-score normalization to all features
        
        Args:
            df_features: DataFrame with article-level features including 'date' and 'category' columns
            window_size: Rolling window size in days for normalization (default: 90)
            min_periods: Minimum periods for rolling calculation (default: 30)
            
        Returns:
            DataFrame with daily features (overall + by category) and normalized versions
            
        Example:
            >>> analyzer = FeatureAnalyzer()
            >>> final_features = analyzer.aggregate_and_normalize_features(df_features)
        """
        print("="*60)
        print("Aggregating and normalizing features")
        print("="*60)
        
        # Step 1: Aggregate to daily frequency (overall)
        daily_features = self.aggregate_features_to_daily(df_features)
        
        # Step 2: Aggregate to daily frequency by category
        daily_features_by_category = self.aggregate_features_by_category_to_daily(df_features)
        
        # Step 3: Pivot category features to category_feature_name format
        print("\nPreparing category features for merge...")
        
        # Feature columns to pivot (exclude date and category)
        feature_cols = ['sentiment_score', 'complexity', 'sentiment_ratio', 
                        'headline_count', 'average_token_length']
        feature_cols = [col for col in feature_cols if col in daily_features_by_category.columns]
        
        if len(feature_cols) == 0:
            category_pivoted = pd.DataFrame({'date': daily_features_by_category['date'].unique()})
        else:
            # Pivot: date as index, category as columns, values as features
            # This creates MultiIndex columns: (feature, category)
            category_pivoted = daily_features_by_category.pivot_table(
                index='date',
                columns='category',
                values=feature_cols,
                aggfunc='first'
            )
            
            # Flatten MultiIndex to category_feature_name format
            # MultiIndex order is (feature, category), so we reverse to get category_feature
            if isinstance(category_pivoted.columns, pd.MultiIndex):
                category_pivoted.columns = [f'{col[1]}_{col[0]}' for col in category_pivoted.columns]
            category_pivoted = category_pivoted.reset_index()
        
        # Step 4: Merge with daily features on date
        merged_features = daily_features.merge(
            category_pivoted,
            on='date',
            how='outer'
        )
        
        # Sort by date
        merged_features = merged_features.sort_values('date').reset_index(drop=True)
        
        print(f"Merged features: {len(merged_features):,} rows, {len(merged_features.columns)} columns")
        
        # Step 5: Normalize all features
        # Get all feature columns (original daily + category-specific)
        all_feature_cols = [col for col in merged_features.columns 
                           if col != 'date' and not col.endswith('_normalized')]
        
        normalized_features = self.normalize_features_rolling_zscore(
            merged_features,
            feature_cols=all_feature_cols,
            window_size=window_size,
            min_periods=min_periods
        )
        
        print("\n" + "="*60)
        print("Feature aggregation and normalization complete")
        print("="*60)

        # Clip normalized features at 1st and 99th percentile to handle outliers
        # Only clip the normalized feature columns (not date column)
        normalized_cols = [col for col in normalized_features.columns 
                          if col.endswith('_normalized')]
        
        if len(normalized_cols) > 0:
            # Get quantiles for normalized columns
            lower_bounds = normalized_features[normalized_cols].quantile(0.01, axis=0)
            upper_bounds = normalized_features[normalized_cols].quantile(0.99, axis=0)
            
            # Clip each normalized column individually
            for col in normalized_cols:
                normalized_features[col] = normalized_features[col].clip(
                    lower=lower_bounds[col], 
                    upper=upper_bounds[col]
                )
            
            print(f"Clipped {len(normalized_cols)} normalized features at 1st and 99th percentiles")

        return normalized_features

    def align_features_with_spy(self, daily_features, df_spy):
        """Align daily features with SPY returns using one-day lag
        
        This function merges features (from aggregate_and_normalize_features) with SPY returns
        and applies a one-day lag to features to avoid look-ahead bias. Features from day t-1
        are used to predict returns on day t.
        
        Args:
            daily_features: DataFrame with daily features (from aggregate_and_normalize_features)
                           Must include 'date' column
            df_spy: DataFrame with SPY returns data. Must include 'date', 'spy_return', 
                   and 'spy_return_next' columns
                   
        Returns:
            DataFrame with features aligned with SPY returns. All feature columns are lagged
            by one day (suffix '_lag' added). Original feature columns are kept.
            
        Example:
            >>> analyzer = FeatureAnalyzer()
            >>> features = analyzer.aggregate_and_normalize_features(df_features)
            >>> aligned = analyzer.align_features_with_spy(features, df_spy)
        """
        print("Aligning features with SPY returns...")
        
        # Ensure dates are datetime type
        if not pd.api.types.is_datetime64_any_dtype(daily_features['date']):
            daily_features = daily_features.copy()
            daily_features['date'] = pd.to_datetime(daily_features['date'])
        if not pd.api.types.is_datetime64_any_dtype(df_spy['date']):
            df_spy = df_spy.copy()
            df_spy['date'] = pd.to_datetime(df_spy['date'])
        
        # Sort both dataframes by date
        daily_features = daily_features.sort_values('date').reset_index(drop=True)
        df_spy = df_spy.sort_values('date').reset_index(drop=True)
        
        # Merge features with SPY data on date
        df_merged = daily_features.merge(
            df_spy[['date', 'spy_return', 'spy_return_next']], 
            on='date', 
            how='inner'
        )
        
        # Apply one-day lag: shift features forward by 1 day
        # This means features from day t-1 are used to predict returns on day t
        feature_cols = [col for col in df_merged.columns 
                       if col not in ['date', 'spy_return', 'spy_return_next']]
        
        print(f"Applying one-day lag to {len(feature_cols)} feature columns...")
        for col in feature_cols:
            df_merged[f'{col}_lag'] = df_merged[col].shift(1)
        
        # Keep only rows where we have both features (from previous day) and returns
        lag_cols = [col for col in df_merged.columns if col.endswith('_lag')]
        df_merged = df_merged[
            (df_merged[lag_cols].notna().any(axis=1)) &
            (df_merged['spy_return_next'].notna())
        ].copy()
        
        # Sort by date
        df_merged = df_merged.sort_values('date').reset_index(drop=True)
        
        print(f"Final merged dataset: {len(df_merged):,} observations")
        print(f"Date range: {df_merged['date'].min()} to {df_merged['date'].max()}")
        print(f"Features aligned: {len(feature_cols)} original + {len(lag_cols)} lagged")
        
        return df_merged
    
    def prepare_features_for_modeling(self, df_features, df_spy, window_size=90, min_periods=30):
        """Complete pipeline: aggregate, normalize, align with SPY, and clean features
        
        This function performs the complete feature engineering pipeline:
        1. Aggregates features to daily frequency (overall + by category)
        2. Applies rolling window z-score normalization
        3. Aligns features with SPY returns using one-day lag
        4. Keeps only normalized lagged features
        5. Removes '_normalized_lag' suffix from feature names
        
        Args:
            df_features: DataFrame with article-level features including 'date' and 'category' columns
            df_spy: DataFrame with SPY returns data. Must include 'date', 'spy_return', 
                   and 'spy_return_next' columns
            window_size: Rolling window size in days for normalization (default: 90)
            min_periods: Minimum periods for rolling calculation (default: 30)
            
        Returns:
            DataFrame with cleaned features ready for modeling. Contains:
            - 'date', 'spy_return', 'spy_return_next' columns
            - Feature columns with clean names (without '_normalized_lag' suffix)
            
        Example:
            >>> analyzer = FeatureAnalyzer()
            >>> df_clean = analyzer.prepare_features_for_modeling(df_features, df_spy)
        """
        print("="*60)
        print("COMPLETE FEATURE ENGINEERING PIPELINE")
        print("="*60)
        
        # Step 1: Aggregate and normalize features
        daily_features = self.aggregate_and_normalize_features(
            df_features,
            window_size=window_size,
            min_periods=min_periods
        )
        
        # Step 2: Align features with SPY returns
        df_merged = self.align_features_with_spy(daily_features, df_spy)
        
        # Step 3: Keep only *_normalized_lag features and remove suffix
        print("\nCleaning features for modeling...")
        normalized_lag_cols = [col for col in df_merged.columns if col.endswith('_normalized_lag')]
        
        if len(normalized_lag_cols) == 0:
            print("Warning: No normalized lagged features found!")
            return df_merged[['date', 'spy_return', 'spy_return_next']]
        
        # Create dataframe with only normalized lagged features
        df_clean = df_merged[['date', 'spy_return', 'spy_return_next'] + normalized_lag_cols].copy()
        
        # Remove '_normalized_lag' suffix from feature column names
        rename_dict = {col: col.replace('_normalized_lag', '') for col in normalized_lag_cols}
        df_clean = df_clean.rename(columns=rename_dict)
        
        print(f"Kept {len(normalized_lag_cols)} normalized lagged features")
        print(f"Final dataset shape: {df_clean.shape}")
        print(f"Feature columns: {len(normalized_lag_cols)}")
        
        print("\n" + "="*60)
        print("Feature engineering pipeline complete")
        print("="*60)
        
        return df_clean
    
    def compute_correlation(self, df, feature_col, target_col='spy_return_next'):
        """Compute Pearson and Spearman correlation between feature and target
        
        Args:
            df: DataFrame with features and target
            feature_col: Name of feature column
            target_col: Name of target column (default: 'spy_return_next')
            
        Returns:
            dict with pearson_r, pearson_p, spearman_r, spearman_p
        """
        # Remove NaN values
        mask = df[[feature_col, target_col]].notna().all(axis=1)
        x = df.loc[mask, feature_col].values
        y = df.loc[mask, target_col].values
        
        if len(x) == 0:
            return {'pearson_r': np.nan, 'pearson_p': np.nan, 
                   'spearman_r': np.nan, 'spearman_p': np.nan}
        
        pearson_r, pearson_p = pearsonr(x, y)
        spearman_r, spearman_p = spearmanr(x, y)
        
        return {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p
        }
    
    def compute_long_short_returns(self, df, feature_col, target_col='spy_return_next', 
                                   quantile_threshold=0.5):
        """Compute long-short portfolio returns based on feature
        
        Args:
            df: DataFrame with features and target
            feature_col: Name of feature column
            target_col: Name of target column (default: 'spy_return_next')
            quantile_threshold: Quantile threshold for long/short split (default: 0.5 = median)
            
        Returns:
            dict with long_returns, short_returns, long_short_spread, and statistics
        """
        # Remove NaN values
        mask = df[[feature_col, target_col]].notna().all(axis=1)
        df_clean = df.loc[mask, [feature_col, target_col]].copy()
        
        if len(df_clean) == 0:
            return {
                'long_returns': [],
                'short_returns': [],
                'long_short_spread': np.nan,
                'long_mean': np.nan,
                'short_mean': np.nan,
                'long_std': np.nan,
                'short_std': np.nan,
                't_stat': np.nan,
                'sharpe_ratio': np.nan
            }
        
        # Split into long and short based on feature quantile
        threshold = df_clean[feature_col].quantile(quantile_threshold)
        long_mask = df_clean[feature_col] >= threshold
        short_mask = df_clean[feature_col] < threshold
        
        long_returns = df_clean.loc[long_mask, target_col].values
        short_returns = df_clean.loc[short_mask, target_col].values
        
        if len(long_returns) == 0 or len(short_returns) == 0:
            return {
                'long_returns': long_returns.tolist(),
                'short_returns': short_returns.tolist(),
                'long_short_spread': np.nan,
                'long_mean': np.nan,
                'short_mean': np.nan,
                'long_std': np.nan,
                'short_std': np.nan,
                't_stat': np.nan,
                'sharpe_ratio': np.nan
            }
        
        long_mean = np.mean(long_returns)
        short_mean = np.mean(short_returns)
        long_std = np.std(long_returns)
        short_std = np.std(short_returns)
        
        long_short_spread = long_mean - short_mean
        
        # T-statistic for difference in means
        n_long = len(long_returns)
        n_short = len(short_returns)
        pooled_std = np.sqrt(((n_long - 1) * long_std**2 + (n_short - 1) * short_std**2) / (n_long + n_short - 2))
        se_diff = pooled_std * np.sqrt(1/n_long + 1/n_short)
        t_stat = long_short_spread / se_diff if se_diff > 0 else np.nan
        
        # Sharpe ratio (annualized)
        # Compute portfolio return series: long when feature >= threshold, short when feature < threshold
        # This creates a matched portfolio return series from the same time periods
        portfolio_returns = df_clean[target_col].copy()
        portfolio_returns[short_mask] = -portfolio_returns[short_mask]  # Short positions: negative returns
        # Long positions: positive returns (already positive)
        
        portfolio_mean = np.mean(portfolio_returns)
        portfolio_std = np.std(portfolio_returns)
        sharpe_ratio = (portfolio_mean / portfolio_std * np.sqrt(252)) if portfolio_std > 0 else np.nan
        
        return {
            'long_returns': long_returns.tolist(),
            'short_returns': short_returns.tolist(),
            'long_short_spread': long_short_spread,
            'long_mean': long_mean,
            'short_mean': short_mean,
            'long_std': long_std,
            'short_std': short_std,
            't_stat': t_stat,
            'sharpe_ratio': sharpe_ratio,
            'n_long': n_long,
            'n_short': n_short
        }
    
    def analyze_feature(self, df, feature_col, target_col='spy_return_next'):
        """Comprehensive analysis of a single feature
        
        Args:
            df: DataFrame with features and target
            feature_col: Name of feature column
            target_col: Name of target column (default: 'spy_return_next')
            
        Returns:
            dict with all analysis results
        """
        results = {}
        
        # Correlation analysis
        corr_results = self.compute_correlation(df, feature_col, target_col)
        results.update(corr_results)
        
        # Long-short analysis
        ls_results = self.compute_long_short_returns(df, feature_col, target_col)
        results.update(ls_results)
        
        return results
    
    def analyze_all_features(self, df, feature_cols=None, target_col='spy_return_next'):
        """Analyze multiple features and return summary DataFrame
        
        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names (if None, auto-detect lagged features)
            target_col: Name of target column (default: 'spy_return_next')
            
        Returns:
            DataFrame with analysis results for each feature
        """
        if feature_cols is None:
            # Auto-detect lagged feature columns
            feature_cols = [col for col in df.columns if col not in ['date', 'spy_return', 'spy_return_next']]
        
        results_list = []
        for feature in feature_cols:
            try:
                analysis = self.analyze_feature(df, feature, target_col)
                analysis['feature'] = feature
                results_list.append(analysis)
            except Exception as e:
                print(f"Error analyzing {feature}: {e}")
                continue
        
        results_df = pd.DataFrame(results_list)
        
        # Reorder columns
        if len(results_df) > 0:
            cols = ['feature', 'pearson_r', 'pearson_p', 'spearman_r', 'spearman_p', 
                   'long_short_spread', 'long_mean', 'short_mean',
                   't_stat', 'sharpe_ratio', 'n_long', 'n_short']
            cols = [c for c in cols if c in results_df.columns]
            results_df = results_df[cols]
        
        return results_df
    
    def plot_feature_analysis(self, df, feature_col, target_col='spy_return_next', 
                             figsize=(15, 10)):
        """Create comprehensive visualization for a single feature
        
        Args:
            df: DataFrame with features and target
            feature_col: Name of feature column
            target_col: Name of target column (default: 'spy_return_next')
            figsize: Figure size tuple
        """
        # Remove NaN values
        mask = df[[feature_col, target_col]].notna().all(axis=1)
        df_clean = df.loc[mask, [feature_col, target_col]].copy()
        
        if len(df_clean) == 0:
            print(f"No data available for {feature_col}")
            return
        
        # Compute analysis
        analysis = self.analyze_feature(df, feature_col, target_col)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Feature Analysis: {feature_col}', fontsize=16, fontweight='bold')
        
        # Plot 1: Scatter plot
        axes[0, 0].scatter(df_clean[feature_col], df_clean[target_col], 
                          alpha=0.3, s=10)
        axes[0, 0].set_xlabel(f'{feature_col}')
        axes[0, 0].set_ylabel(target_col)
        axes[0, 0].set_title(f'Scatter Plot\nPearson r={analysis["pearson_r"]:.4f}, '
                            f'Spearman r={analysis["spearman_r"]:.4f}')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add regression line
        z = np.polyfit(df_clean[feature_col], df_clean[target_col], 1)
        p = np.poly1d(z)
        axes[0, 0].plot(df_clean[feature_col], p(df_clean[feature_col]), 
                       "r--", alpha=0.8, linewidth=2)
        
        # Plot 2: Long-short returns distribution
        ls_results = self.compute_long_short_returns(df, feature_col, target_col)
        if len(ls_results['long_returns']) > 0 and len(ls_results['short_returns']) > 0:
            axes[0, 1].hist(ls_results['long_returns'], bins=50, alpha=0.6, 
                           label=f'Long (mean={ls_results["long_mean"]:.4f})', color='green')
            axes[0, 1].hist(ls_results['short_returns'], bins=50, alpha=0.6, 
                           label=f'Short (mean={ls_results["short_mean"]:.4f})', color='red')
            axes[0, 1].set_xlabel('Returns')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title(f'Long-Short Returns Distribution\n'
                                f'Spread={ls_results["long_short_spread"]:.4f}, '
                                f'Sharpe={ls_results["sharpe_ratio"]:.2f}')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Feature time series
        if 'date' in df.columns:
            df_time = df[['date', feature_col]].copy()
            df_time = df_time.sort_values('date')
            axes[1, 0].plot(df_time['date'], df_time[feature_col], linewidth=1, alpha=0.7)
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel(f'{feature_col}')
            axes[1, 0].set_title('Feature Time Series')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Summary statistics table
        axes[1, 1].axis('off')
        stats_text = f"""
        Correlation Analysis:
        Pearson r: {analysis['pearson_r']:.4f} (p={analysis['pearson_p']:.4f})
        Spearman r: {analysis['spearman_r']:.4f} (p={analysis['spearman_p']:.4f})
        
        Long-Short Analysis:
        Long Mean: {ls_results['long_mean']:.4f}
        Short Mean: {ls_results['short_mean']:.4f}
        Spread: {ls_results['long_short_spread']:.4f}
        T-statistic: {ls_results['t_stat']:.2f}
        Sharpe Ratio: {ls_results['sharpe_ratio']:.2f}
        N (Long): {ls_results['n_long']}
        N (Short): {ls_results['n_short']}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                       verticalalignment='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.show()
    
    def plot_top_features(self, results_df, measure='sharpe_ratio', top_n=15, 
                         figsize=(14, 10), ascending=False):
        """Visualize top and bottom N features sorted by a specified measure
        
        Shows both top N (highest) and bottom N (lowest) features. If bottom values
        are all 0 or top values are all 0, only shows the non-zero group.
        
        Args:
            results_df: DataFrame from analyze_all_features() with feature analysis results
            measure: Column name to sort by (e.g., 'sharpe_ratio', 'pearson_r', 'spearman_r', 't_stat')
            top_n: Number of top and bottom features to display (default: 15)
            figsize: Figure size tuple (default: (14, 10))
            ascending: Whether to sort in ascending order (default: False for descending)
                      Set to True for p-values or negative metrics
            
        Example:
            >>> analyzer = FeatureAnalyzer()
            >>> results = analyzer.analyze_all_features(df, feature_cols)
            >>> analyzer.plot_top_features(results, measure='sharpe_ratio', top_n=15)
        """
        if measure not in results_df.columns:
            raise ValueError(f"Measure '{measure}' not found in results DataFrame. "
                           f"Available columns: {list(results_df.columns)}")
        
        # Create a copy and remove rows with NaN values for the measure
        df_plot = results_df.copy()
        df_plot = df_plot[df_plot[measure].notna()].copy()
        
        if len(df_plot) == 0:
            print(f"No valid data found for measure '{measure}'")
            return
        
        # Sort by measure
        df_sorted = df_plot.sort_values(measure, ascending=ascending)
        
        # Get top N and bottom N
        df_top = df_sorted.head(top_n).copy()
        df_bottom = df_sorted.tail(top_n).copy()
        
        # Check if we need to filter zeros
        top_has_nonzero = (df_top[measure] != 0).any() if len(df_top) > 0 else False
        bottom_has_nonzero = (df_bottom[measure] != 0).any() if len(df_bottom) > 0 else False
        
        # Filter out zeros if needed
        if top_has_nonzero and not bottom_has_nonzero:
            # Only show top if bottom is all zeros
            df_top = df_top[df_top[measure] != 0].head(top_n)
            df_bottom = pd.DataFrame()
        elif bottom_has_nonzero and not top_has_nonzero:
            # Only show bottom if top is all zeros
            df_bottom = df_bottom[df_bottom[measure] != 0].head(top_n)
            df_top = pd.DataFrame()
        else:
            # Both have non-zero values, filter zeros from each
            df_top = df_top[df_top[measure] != 0].head(top_n)
            df_bottom = df_bottom[df_bottom[measure] != 0].head(top_n)
        
        # Determine number of subplots needed
        n_plots = 0
        if len(df_top) > 0:
            n_plots += 1
        if len(df_bottom) > 0:
            n_plots += 1
        
        if n_plots == 0:
            print(f"All values for measure '{measure}' are zero. Nothing to plot.")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        if n_plots == 1:
            axes = [axes]  # Make it iterable
        
        plot_idx = 0
        
        # Plot top features
        if len(df_top) > 0:
            ax = axes[plot_idx]
            y_pos = np.arange(len(df_top))
            colors = plt.cm.viridis(np.linspace(0, 1, len(df_top)))
            
            bars = ax.barh(y_pos, df_top[measure].values, color=colors)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(df_top['feature'].values, fontsize=9)
            ax.set_xlabel(measure.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.set_title(f'Top {len(df_top)} Features by {measure.replace("_", " ").title()}', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels on bars
            for i, (idx, row) in enumerate(df_top.iterrows()):
                value = row[measure]
                ax.text(value, i, f' {value:.4f}', 
                       va='center', fontsize=8, fontweight='bold')
            
            ax.invert_yaxis()
            plot_idx += 1
        
        # Plot bottom features
        if len(df_bottom) > 0:
            ax = axes[plot_idx]
            y_pos = np.arange(len(df_bottom))
            colors = plt.cm.plasma(np.linspace(0, 1, len(df_bottom)))
            
            bars = ax.barh(y_pos, df_bottom[measure].values, color=colors)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(df_bottom['feature'].values, fontsize=9)
            ax.set_xlabel(measure.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.set_title(f'Bottom {len(df_bottom)} Features by {measure.replace("_", " ").title()}', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels on bars
            for i, (idx, row) in enumerate(df_bottom.iterrows()):
                value = row[measure]
                ax.text(value, i, f' {value:.4f}', 
                       va='center', fontsize=8, fontweight='bold')
            
            ax.invert_yaxis()
        
        plt.tight_layout()
        plt.show()
        
        return fig, axes
    
    def analyze_pairwise_correlations(self, df, feature_cols=None, method='pearson', 
                                      figsize=(16, 14), top_pairs=20):
        """Analyze pairwise correlations between features
        
        Computes correlation matrix for all features and visualizes it as a heatmap.
        Also identifies and displays the most highly correlated feature pairs.
        
        Args:
            df: DataFrame with features
            feature_cols: List of feature column names (if None, auto-detect)
            method: Correlation method ('pearson', 'spearman', or 'kendall')
            figsize: Figure size tuple for heatmap (default: (16, 14))
            top_pairs: Number of top correlated pairs to display (default: 20)
            
        Returns:
            tuple: (correlation_matrix, top_correlations_df)
                - correlation_matrix: DataFrame with pairwise correlations
                - top_correlations_df: DataFrame with top correlated pairs
                
        Example:
            >>> analyzer = FeatureAnalyzer()
            >>> corr_matrix, top_pairs = analyzer.analyze_pairwise_correlations(df_clean)
        """
        # Get feature columns
        if feature_cols is None:
            feature_cols = [col for col in df.columns 
                           if col not in ['date', 'spy_return', 'spy_return_next']]
        
        # Extract feature data
        feature_data = df[feature_cols].copy()
        
        # Remove columns with all NaN or constant values
        feature_data = feature_data.loc[:, feature_data.nunique() > 1]
        feature_cols = [col for col in feature_cols if col in feature_data.columns]
        
        if len(feature_cols) < 2:
            print("Need at least 2 features for pairwise correlation analysis")
            return None, None
        
        print(f"Computing {method} pairwise correlations for {len(feature_cols)} features...")
        
        # Compute correlation matrix
        corr_matrix = feature_data[feature_cols].corr(method=method)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(figsize[0], figsize[1] // 2))
        
        # Plot 1: Full correlation heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=axes[0],
                   xticklabels=False, yticklabels=False)
        axes[0].set_title(f'Feature Pairwise Correlation Matrix ({method.title()})', 
                         fontsize=14, fontweight='bold')
        
        # Plot 2: Top correlated pairs
        # Get upper triangle of correlation matrix (excluding diagonal)
        corr_matrix_upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Flatten and sort
        corr_pairs = []
        for i in range(len(corr_matrix_upper.columns)):
            for j in range(i+1, len(corr_matrix_upper.columns)):
                feat1 = corr_matrix_upper.columns[i]
                feat2 = corr_matrix_upper.columns[j]
                corr_val = corr_matrix_upper.iloc[i, j]
                if not np.isnan(corr_val):
                    corr_pairs.append({
                        'feature_1': feat1,
                        'feature_2': feat2,
                        'correlation': corr_val,
                        'abs_correlation': abs(corr_val)
                    })
        
        top_correlations_df = pd.DataFrame(corr_pairs)
        top_correlations_df = top_correlations_df.sort_values('abs_correlation', ascending=False).head(top_pairs)
        
        # Plot top pairs
        y_pos = np.arange(len(top_correlations_df))
        colors = ['red' if x < 0 else 'blue' for x in top_correlations_df['correlation'].values]
        
        axes[1].barh(y_pos, top_correlations_df['correlation'].values, color=colors, alpha=0.7)
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(
            [f"{row['feature_1']} vs {row['feature_2']}" 
             for _, row in top_correlations_df.iterrows()],
            fontsize=8
        )
        axes[1].set_xlabel(f'{method.title()} Correlation', fontsize=12, fontweight='bold')
        axes[1].set_title(f'Top {len(top_correlations_df)} Correlated Feature Pairs', 
                         fontsize=14, fontweight='bold')
        axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].grid(True, alpha=0.3, axis='x')
        axes[1].invert_yaxis()
        
        # Add value labels
        for i, (idx, row) in enumerate(top_correlations_df.iterrows()):
            value = row['correlation']
            axes[1].text(value, i, f' {value:.3f}', 
                        va='center', fontsize=7, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"\nCorrelation Summary:")
        print(f"  Total feature pairs: {len(corr_pairs)}")
        print(f"  Mean absolute correlation: {top_correlations_df['abs_correlation'].mean():.4f}")
        print(f"  Max absolute correlation: {top_correlations_df['abs_correlation'].max():.4f}")
        print(f"  Min absolute correlation: {top_correlations_df['abs_correlation'].min():.4f}")
        print(f"\nTop {top_pairs} correlated pairs:")
        print(top_correlations_df[['feature_1', 'feature_2', 'correlation']].to_string(index=False))
        
        return corr_matrix, top_correlations_df
    
    def analyze_feature_distributions(self, df, feature_cols=None, sample_size=12, 
                                     figsize=(16, 10)):
        """Analyze feature distributions with summary statistics and visualizations
        
        Provides a comprehensive analysis of feature distributions including:
        - Summary statistics table
        - Sample histogram grid (showing subset of features)
        - Box plot summary
        
        Args:
            df: DataFrame with features
            feature_cols: List of feature column names (if None, auto-detect)
            sample_size: Number of features to show in histogram grid (default: 12)
            figsize: Figure size tuple (default: (16, 10))
            
        Returns:
            DataFrame: Summary statistics for all features
            
        Example:
            >>> analyzer = FeatureAnalyzer()
            >>> stats_df = analyzer.analyze_feature_distributions(df_clean)
        """
        
        # Get feature columns
        if feature_cols is None:
            feature_cols = [col for col in df.columns 
                           if col not in ['date', 'spy_return', 'spy_return_next']]
        
        # Extract feature data
        feature_data = df[feature_cols].copy()
        
        # Remove columns with all NaN or constant values
        feature_data = feature_data.loc[:, feature_data.nunique() > 1]
        feature_cols = [col for col in feature_cols if col in feature_data.columns]
        
        if len(feature_cols) == 0:
            print("No valid features found for distribution analysis")
            return None
        
        print(f"Analyzing distributions for {len(feature_cols)} features...")
        
        # Calculate summary statistics
        stats_list = []
        for col in feature_cols:
            values = feature_data[col].dropna()
            if len(values) > 0:
                stats_list.append({
                    'feature': col,
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75),
                    'skewness': skew(values),
                    'kurtosis': kurtosis(values),
                    'missing_pct': (df[col].isna().sum() / len(df)) * 100
                })
        
        stats_df = pd.DataFrame(stats_list)
        
        # Create visualizations
        fig = plt.figure(figsize=figsize)
        
        # Select features to visualize (sample by variance or randomly)
        # Select features with highest variance for better visualization
        stats_df_sorted = stats_df.sort_values('std', ascending=False)
        sample_features = stats_df_sorted.head(sample_size)['feature'].tolist()
        
        # Plot 1: Histogram grid for sample features
        n_cols = 4
        n_rows = int(np.ceil(len(sample_features) / n_cols))
        
        for idx, feat in enumerate(sample_features):
            ax = plt.subplot(n_rows, n_cols, idx + 1)
            values = feature_data[feat].dropna()
            if len(values) > 0:
                ax.hist(values, bins=30, alpha=0.7, edgecolor='black')
                ax.set_title(f'{feat}\n(μ={np.mean(values):.3f}, σ={np.std(values):.3f})', 
                           fontsize=9)
                ax.set_xlabel('Value', fontsize=8)
                ax.set_ylabel('Frequency', fontsize=8)
                ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Feature Distribution Histograms (Sample of {len(sample_features)} Features)', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.show()
        
        # Plot 2: Box plot summary (showing distribution spread)
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Prepare data for box plot (select top features by variance)
        box_features = stats_df_sorted.head(min(20, len(stats_df)))['feature'].tolist()
        box_data = [feature_data[feat].dropna().values for feat in box_features]
        
        bp = ax.boxplot(box_data, labels=box_features, vert=True, patch_artist=True)
        
        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Feature Value', fontsize=12, fontweight='bold')
        ax.set_title(f'Feature Distribution Box Plots (Top {len(box_features)} Features by Variance)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.tight_layout()
        plt.show()
        
        # Plot 3: Summary statistics visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Mean vs Std scatter
        axes[0].scatter(stats_df['mean'], stats_df['std'], alpha=0.6, s=50)
        axes[0].set_xlabel('Mean', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Standard Deviation', fontsize=11, fontweight='bold')
        axes[0].set_title('Mean vs Standard Deviation', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Skewness distribution
        axes[1].hist(stats_df['skewness'], bins=30, alpha=0.7, edgecolor='black', color='skyblue')
        axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Normal (skew=0)')
        axes[1].set_xlabel('Skewness', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Number of Features', fontsize=11, fontweight='bold')
        axes[1].set_title('Distribution of Skewness', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('Feature Distribution Summary Statistics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        print("\n" + "="*80)
        print("Distribution analysis complete")
        print("="*80)
        
        return stats_df
    
    def generate_feature_groups(self, feature_cols):
        """Automatically generate feature groups from feature column names
        
        Categorizes features into groups based on naming patterns:
        - sentiment: sentiment_score, sentiment_ratio (including category-specific)
        - complexity: complexity features (including category-specific)
        - token_length: average_token_length, headline_tokens (including category-specific)
        - volume: headline_count features (including category-specific)
        - uncertainty: uncertainty-related features
        
        Args:
            feature_cols: List of feature column names
            
        Returns:
            dict: Mapping of group name -> list of feature names
            
        Example:
            >>> feature_cols = ['sentiment_score', 'complexity', 'POLITICS_sentiment_score', ...]
            >>> groups = analyzer.generate_feature_groups(feature_cols)
            >>> print(groups)
            {'sentiment': [...], 'complexity': [...], 'token_length': [...], ...}
        """
        groups = {
            'sentiment': [],
            'complexity': [],
            'token_length': [],
            'volume': [],
            'uncertainty': []
        }
        
        for feat in feature_cols:
            feat_lower = feat.lower()
            
            # Categorize features (treat category-specific and overall features the same)
            if 'sentiment' in feat_lower:
                groups['sentiment'].append(feat)
            elif 'complexity' in feat_lower:
                groups['complexity'].append(feat)
            elif 'token' in feat_lower or 'length' in feat_lower:
                groups['token_length'].append(feat)
            elif 'count' in feat_lower or 'volume' in feat_lower:
                groups['volume'].append(feat)
            elif 'uncertainty' in feat_lower:
                groups['uncertainty'].append(feat)
            else:
                # If no match, assign to sentiment as default (most common type)
                groups['sentiment'].append(feat)
        
        # Remove empty groups
        groups = {k: v for k, v in groups.items() if len(v) > 0}
        
        # Print summary
        print("="*80)
        print("FEATURE GROUPS GENERATED")
        print("="*80)
        print(f"\nTotal features: {len(feature_cols)}")
        print(f"Number of groups: {len(groups)}")
        print("\nGroup breakdown:")
        for group_name, features in sorted(groups.items()):
            print(f"  {group_name}: {len(features)} features")
        print("="*80)
        
        return groups
    
    def compute_forward_returns(self, df, close_col='spy_return', horizons=[1, 2, 3, 4, 5, 7, 10, 14]):
        """Compute forward log-returns for multiple horizons
        
        Args:
            df: DataFrame with date index and close prices/returns
            close_col: Column name for close prices or returns (default: 'spy_return')
            horizons: List of forward horizons in days (default: [1, 2, 3, 4, 5, 7, 10, 14])
            
        Returns:
            DataFrame: Forward returns with columns ret_1, ret_2, etc.
        """
        df = df.copy()
        df = df.sort_index() if isinstance(df.index, pd.DatetimeIndex) else df.sort_values('date')
        
        forward_returns = {}
        
        for h in horizons:
            if h <= 0:
                continue
            
            # Compute forward return: log(close.shift(-h) / close)
            # For returns, we can use cumulative return: (1 + ret).cumprod()
            if close_col in df.columns:
                if 'return' in close_col.lower():
                    # If it's already returns, compute cumulative forward return
                    forward_ret = np.log((1 + df[close_col]).shift(-h) / (1 + df[close_col]))
                else:
                    # If it's prices, compute log return
                    forward_ret = np.log(df[close_col].shift(-h) / df[close_col])
                
                forward_returns[f'ret_{h}'] = forward_ret
        
        return pd.DataFrame(forward_returns, index=df.index)
    
    def compute_feature_sharpe(self, feature, returns, horizon=1, window=60, min_obs=250):
        """Compute Sharpe ratio for a feature using rolling z-score signal
        
        Args:
            feature: Series of feature values
            returns: Series of forward returns
            horizon: Forward horizon in days for proper annualization (default: 1)
            window: Rolling window for z-score normalization (default: 60)
            min_obs: Minimum observations required (default: 250)
            
        Returns:
            dict: Sharpe ratio and related metrics
        """
        # Align feature and returns
        aligned = pd.DataFrame({
            'feature': feature,
            'return': returns
        }).dropna()
        
        if len(aligned) < min_obs:
            return {
                'sharpe': np.nan,
                'mean_return': np.nan,
                'std_return': np.nan,
                'n_obs': len(aligned)
            }
        
        # Compute rolling z-score signal
        rolling_mean = aligned['feature'].rolling(window=window, min_periods=30).mean()
        rolling_std = aligned['feature'].rolling(window=window, min_periods=30).std()
        z_signal = (aligned['feature'] - rolling_mean) / rolling_std
        z_signal = z_signal.fillna(0)
        
        # Compute positions: +1 if signal > 0, -1 if signal < 0, 0 if |signal| < 0.25
        positions = pd.Series(0, index=aligned.index)
        positions[z_signal > 0] = 1
        positions[z_signal < 0] = -1
        positions[z_signal.abs() < 0.25] = 0
        
        # Strategy returns
        strategy_returns = positions * aligned['return']
        strategy_returns = strategy_returns.dropna()
        
        if len(strategy_returns) < min_obs:
            return {
                'sharpe': np.nan,
                'mean_return': np.nan,
                'std_return': np.nan,
                'n_obs': len(strategy_returns)
            }
        
        # Annualize Sharpe
        # For h-day forward returns, annualization factors:
        # Mean: multiply by (252/h) to get annualized mean
        # Std: multiply by sqrt(252/h) to get annualized std
        mean_ret = strategy_returns.mean()
        std_ret = strategy_returns.std()
        
        # Annualize based on horizon
        annualized_mean = mean_ret * (252 / horizon)
        annualized_std = std_ret * np.sqrt(252 / horizon)
        
        sharpe = annualized_mean / annualized_std if annualized_std > 0 else np.nan
        
        return {
            'sharpe': sharpe,
            'mean_return': mean_ret,
            'std_return': std_ret,
            'n_obs': len(strategy_returns)
        }
    
    def compute_feature_correlations(self, feature, returns):
        """Compute Spearman and Pearson correlations for a feature
        
        Args:
            feature: Series of feature values
            returns: Series of forward returns
            
        Returns:
            dict: Correlation metrics
        """
        # Align feature and returns
        aligned = pd.DataFrame({
            'feature': feature,
            'return': returns
        }).dropna()
        
        if len(aligned) < 30:
            return {
                'spearman_r': np.nan,
                'spearman_p': np.nan,
                'pearson_r': np.nan,
                'pearson_p': np.nan,
                'n_obs': len(aligned)
            }
        
        # Compute correlations
        spearman_r, spearman_p = spearmanr(aligned['feature'], aligned['return'])
        pearson_r, pearson_p = pearsonr(aligned['feature'], aligned['return'])
        
        return {
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'n_obs': len(aligned)
        }
    
    def analyze_horizon_selection(self, df, feature_groups, horizons=[1, 2, 3, 4, 5, 7, 10, 14],
                                 close_col='spy_return_next', window=60, min_obs=250):
        """Analyze optimal prediction horizon using Sharpe-first priority
        
        Args:
            df: DataFrame with features and target returns
            feature_groups: Dict mapping group name -> list of feature columns, OR
                           list of feature column names (will be grouped as 'all_features')
            close_col: Column name for forward returns (default: 'spy_return_next')
            horizons: List of forward horizons in days (default: [1, 2, 3, 4, 5, 7, 10, 14])
            window: Rolling window for z-score (default: 60)
            min_obs: Minimum observations required (default: 250)
            
        Returns:
            dict: Analysis results including feature metrics, group summaries, and rankings
        """
        print("="*80)
        print("HORIZON SELECTION ANALYSIS (Sharpe-First Priority)")
        print("="*80)
        
        # Handle both dict and list inputs for feature_groups
        if isinstance(feature_groups, list):
            # If list provided, create a single group called 'all_features'
            feature_groups = {'all_features': feature_groups}
        elif not isinstance(feature_groups, dict):
            raise ValueError("feature_groups must be either a dict or a list of feature names")
        
        # Ensure horizons don't exceed 14 days
        horizons = [h for h in horizons if h <= 14]
        horizons = sorted(horizons)
        
        print(f"\nAnalyzing horizons: {horizons} days")
        print(f"Feature groups: {list(feature_groups.keys())}")
        
        # Step 1: Compute forward returns for each horizon
        print("\nStep 1: Computing forward returns...")
        forward_returns = {}
        
        # For each horizon, we need to compute forward cumulative returns
        # Since we have spy_return_next, we'll create forward returns
        df_work = df.copy()
        
        if 'date' in df_work.columns:
            df_work = df_work.set_index('date')
        
        df_work = df_work.sort_index()
        
        # Create forward returns for each horizon
        # Since spy_return_next is already 1-day forward return, we need to compute
        # cumulative returns over h days by compounding
        for h in horizons:
            if close_col in df_work.columns:
                if h == 1:
                    forward_returns[f'ret_{h}'] = df_work[close_col]
                else:
                    # Cumulative return over h days: (1+r1)*(1+r2)*...*(1+rh) - 1
                    # Convert to log returns for easier computation
                    cumulative_ret = np.log(1 + df_work[close_col]).copy()
                    for i in range(1, h):
                        cumulative_ret = cumulative_ret + np.log(1 + df_work[close_col].shift(-i).fillna(0))
                    # Convert back to simple returns
                    forward_returns[f'ret_{h}'] = np.exp(cumulative_ret) - 1
        
        forward_returns_df = pd.DataFrame(forward_returns, index=df_work.index)
        
        # Step 2: Compute feature-level metrics for each (feature, horizon)
        print("\nStep 2: Computing feature-level metrics...")
        feature_metrics_list = []
        
        all_features = []
        for group, features in feature_groups.items():
            all_features.extend([(f, group) for f in features if f in df_work.columns])
        
        print(f"Total features to analyze: {len(all_features)}")
        
        for feature_name, group in all_features:
            feature_series = df_work[feature_name]
            
            for h in horizons:
                ret_col = f'ret_{h}'
                if ret_col not in forward_returns_df.columns:
                    continue
                
                returns_series = forward_returns_df[ret_col]
                
                # Compute Sharpe
                sharpe_result = self.compute_feature_sharpe(
                    feature_series, returns_series, horizon=h, window=window, min_obs=min_obs
                )
                
                # Compute correlations
                corr_result = self.compute_feature_correlations(feature_series, returns_series)
                
                feature_metrics_list.append({
                    'horizon': h,
                    'feature': feature_name,
                    'group': group,
                    'sharpe': sharpe_result['sharpe'],
                    'spearman_r': corr_result['spearman_r'],
                    'pearson_r': corr_result['pearson_r'],
                    'n_obs': max(sharpe_result['n_obs'], corr_result['n_obs'])
                })
        
        df_feature_metrics = pd.DataFrame(feature_metrics_list)
        
        print(f"Computed metrics for {len(df_feature_metrics)} feature-horizon combinations")
        
        # Step 3: Group-level aggregation (Sharpe-first)
        print("\nStep 3: Aggregating by group...")
        group_summary_list = []
        
        for group in feature_groups.keys():
            for h in horizons:
                group_data = df_feature_metrics[
                    (df_feature_metrics['group'] == group) & 
                    (df_feature_metrics['horizon'] == h)
                ]
                
                if len(group_data) == 0:
                    continue
                
                # Sharpe metrics (PRIMARY)
                sharpe_vals = group_data['sharpe'].dropna()
                median_sharpe = sharpe_vals.median()
                top_quartile_sharpe = sharpe_vals.quantile(0.75)
                max_sharpe = sharpe_vals.max()
                count_sharpe_pos = (sharpe_vals > 0).sum()
                
                # Spearman metrics (SECONDARY)
                spearman_vals = group_data['spearman_r'].dropna()
                median_spearman = spearman_vals.median()
                top_quartile_spearman = spearman_vals.quantile(0.75)
                
                # Pearson metrics (TERTIARY)
                pearson_vals = group_data['pearson_r'].dropna()
                median_pearson = pearson_vals.median()
                
                group_summary_list.append({
                    'horizon': h,
                    'group': group,
                    'median_sharpe': median_sharpe,
                    'top_quartile_sharpe': top_quartile_sharpe,
                    'max_sharpe': max_sharpe,
                    'count_sharpe_pos': count_sharpe_pos,
                    'median_spearman': median_spearman,
                    'top_quartile_spearman': top_quartile_spearman,
                    'median_pearson': median_pearson,
                    'n_features': len(group_data)
                })
        
        df_group_summary = pd.DataFrame(group_summary_list)
        
        # Print group-level statistics
        print("\n" + "="*80)
        print("GROUP-LEVEL STATISTICS BY HORIZON")
        print("="*80)
        
        for group in sorted(df_group_summary['group'].unique()):
            group_data = df_group_summary[df_group_summary['group'] == group].sort_values('horizon')
            print(f"\n{group.upper().replace('_', ' ')}:")
            print("-" * 60)
            print(group_data[['horizon', 'median_sharpe', 'top_quartile_sharpe', 'max_sharpe', 
                             'count_sharpe_pos', 'median_spearman', 'top_quartile_spearman', 
                             'median_pearson', 'n_features']].to_string(index=False))
        
        print("\n" + "="*80)
        
        # Step 4: Horizon ranking
        print("\nStep 4: Ranking horizons...")
        horizon_ranking_list = []
        
        for h in horizons:
            horizon_data = df_group_summary[df_group_summary['horizon'] == h]
            
            if len(horizon_data) == 0:
                continue
            
            # Aggregate across all groups
            median_sharpe = horizon_data['median_sharpe'].median()
            top_quartile_sharpe = horizon_data['top_quartile_sharpe'].median()
            median_spearman = horizon_data['median_spearman'].median()
            median_pearson = horizon_data['median_pearson'].median()
            
            horizon_ranking_list.append({
                'horizon': h,
                'median_sharpe': median_sharpe,
                'top_quartile_sharpe': top_quartile_sharpe,
                'median_spearman': median_spearman,
                'median_pearson': median_pearson
            })
        
        df_horizon_ranking = pd.DataFrame(horizon_ranking_list)
        
        # Compute composite rank (Sharpe-first priority)
        # Rank 1 = best, higher rank = worse
        # NaN values are ranked last (worst) by default
        df_horizon_ranking['rank_median_sharpe'] = df_horizon_ranking['median_sharpe'].rank(ascending=False, na_option='bottom')
        df_horizon_ranking['rank_top_quartile_sharpe'] = df_horizon_ranking['top_quartile_sharpe'].rank(ascending=False, na_option='bottom')
        df_horizon_ranking['rank_median_spearman'] = df_horizon_ranking['median_spearman'].abs().rank(ascending=False, na_option='bottom')
        df_horizon_ranking['rank_median_pearson'] = df_horizon_ranking['median_pearson'].abs().rank(ascending=False, na_option='bottom')
        
        # Composite rank: weighted sum with Sharpe priority
        df_horizon_ranking['composite_rank'] = (
            0.5 * df_horizon_ranking['rank_median_sharpe'] +
            0.3 * df_horizon_ranking['rank_top_quartile_sharpe'] +
            0.15 * df_horizon_ranking['rank_median_spearman'] +
            0.05 * df_horizon_ranking['rank_median_pearson']
        )
        
        df_horizon_ranking = df_horizon_ranking.sort_values('composite_rank')
        
        print("\n" + "="*80)
        print("HORIZON RANKING (Sharpe-First Priority)")
        print("="*80)
        print(df_horizon_ranking[['horizon', 'median_sharpe', 'top_quartile_sharpe', 
                                  'median_spearman', 'median_pearson', 'composite_rank']].to_string(index=False))
        
        results = {
            'feature_metrics': df_feature_metrics,
            'group_summary': df_group_summary,
            'horizon_ranking': df_horizon_ranking
        }
        
        return results
    
    def plot_horizon_analysis(self, results, figsize=(16, 12)):
        """Visualize horizon selection analysis with Sharpe-focused plots
        
        Args:
            results: Dict from analyze_horizon_selection()
            figsize: Figure size tuple
            
        Returns:
            fig, axes: Matplotlib figure and axes
        """
        df_group_summary = results['group_summary']
        df_horizon_ranking = results['horizon_ranking']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Horizon Selection Analysis (Sharpe-First Priority)', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Heatmap - Median Sharpe by (group × horizon)
        pivot_sharpe = df_group_summary.pivot(index='group', columns='horizon', values='median_sharpe')
        sns.heatmap(pivot_sharpe, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                   ax=axes[0, 0], cbar_kws={'label': 'Median Sharpe'})
        axes[0, 0].set_title('Median Sharpe Ratio by Group × Horizon', fontweight='bold')
        axes[0, 0].set_xlabel('Horizon (days)')
        axes[0, 0].set_ylabel('Feature Group')
        
        # Plot 2: Line plot - Median Sharpe vs Horizon (one line per group)
        groups = df_group_summary['group'].unique()
        for group in groups:
            group_data = df_group_summary[df_group_summary['group'] == group].sort_values('horizon')
            axes[0, 1].plot(group_data['horizon'], group_data['median_sharpe'], 
                          marker='o', label=group, linewidth=2)
        axes[0, 1].set_title('Median Sharpe Ratio vs Horizon', fontweight='bold')
        axes[0, 1].set_xlabel('Horizon (days)')
        axes[0, 1].set_ylabel('Median Sharpe Ratio')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        
        # Plot 3: Heatmap - Median Spearman by (group × horizon)
        pivot_spearman = df_group_summary.pivot(index='group', columns='horizon', values='median_spearman')
        sns.heatmap(pivot_spearman, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                   ax=axes[1, 0], cbar_kws={'label': 'Median Spearman'})
        axes[1, 0].set_title('Median Spearman Correlation by Group × Horizon', fontweight='bold')
        axes[1, 0].set_xlabel('Horizon (days)')
        axes[1, 0].set_ylabel('Feature Group')
        
        # Plot 4: Horizon ranking bar chart
        df_rank_sorted = df_horizon_ranking.sort_values('composite_rank')
        axes[1, 1].barh(range(len(df_rank_sorted)), df_rank_sorted['median_sharpe'], 
                       color='skyblue', alpha=0.7)
        axes[1, 1].set_yticks(range(len(df_rank_sorted)))
        axes[1, 1].set_yticklabels([f"{int(h)} days" for h in df_rank_sorted['horizon']])
        axes[1, 1].set_xlabel('Median Sharpe Ratio', fontweight='bold')
        axes[1, 1].set_title('Horizon Ranking by Composite Score', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        axes[1, 1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 1].invert_yaxis()
        
        plt.tight_layout()
        plt.show()
        
        return fig, axes


def run():
    from data_loader import DataLoader
    from feature_extractor import ArticleFeatureExtractor
    data_loader = DataLoader()
    df_news = data_loader.load_news_dataset()
    extractor = ArticleFeatureExtractor(data_loader=data_loader)
    df_features = extractor.compute_all_features(df_news, reload_cache=True)
    feature_analyzer = FeatureAnalyzer()
    df_spy = data_loader.load_spy_returns()
    df_clean = feature_analyzer.prepare_features_for_modeling(df_features, df_spy)
    print(df_clean.head())
    feature_cols = [col for col in df_clean.columns if col not in ['date', 'spy_return', 'spy_return_next']]
    # analyze horizon selection
    horizons = [1, 2, 3, 4, 5, 7, 10, 14]
    # generate feature groups
    feature_groups = feature_analyzer.generate_feature_groups(feature_cols)
    results = feature_analyzer.analyze_horizon_selection(df_clean, feature_groups, horizons)
    feature_analyzer.plot_horizon_analysis(results)
    # use rolling 7 day, 14 day and 30 day z-score to analyze the horizon selection
    for window in [7, 14, 30]:
        df_rolling = df_clean[feature_cols].rolling(window=window).mean()
        df_rolling = df_rolling.merge(df_clean[['date', 'spy_return', 'spy_return_next']], left_index=True, right_index=True)
        results = feature_analyzer.analyze_horizon_selection(df_rolling, feature_groups, horizons)
        feature_analyzer.plot_horizon_analysis(results)
    # plot horizon analysis
    feature_analyzer.plot_horizon_analysis(results)
    # analyze all features
    feature_analysis_results = feature_analyzer.analyze_all_features(df_clean, feature_cols)
    feature_analysis_results.head(20)
    # correlation matrix
    corr_matrix, top_correlations = feature_analyzer.analyze_pairwise_correlations(
        df_clean,
        feature_cols=feature_cols,
        method='pearson',
        figsize=(16, 14),
        top_pairs=20
    )
    # distribution analysis
    stats_df = feature_analyzer.analyze_feature_distributions(df_clean, feature_cols)

