"""
Modeling script for SPY returns prediction using news features
Uses time series cross-validation and tests multiple models
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


class NewsSentimentModeler:
    """Class for modeling SPY returns using news sentiment features"""
    
    def __init__(self):
        """Initialize the modeler"""
        self.models = {}
        self.scalers = {}
        self.cv_results = {}
        self.test_results = {}
        self.oos_results = {}
    
    def split_data_by_date(self, df, train_start='2012-01-01', train_end='2019-12-31',
                          test_start='2020-01-01', test_end='2021-12-31',
                          oos_start='2022-01-01', oos_end='2022-12-31'):
        """Split data into train, test, and out-of-sample sets by date
        
        Args:
            df: DataFrame with 'date' column and features
            train_start: Start date for training set
            train_end: End date for training set
            test_start: Start date for test set
            test_end: End date for test set
            oos_start: Start date for out-of-sample set
            oos_end: End date for out-of-sample set
            
        Returns:
            tuple: (df_train, df_test, df_oos)
        """
        # Ensure date column is datetime
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Split by date ranges
        df_train = df[(df['date'] >= train_start) & (df['date'] <= train_end)].copy()
        df_test = df[(df['date'] >= test_start) & (df['date'] <= test_end)].copy()
        df_oos = df[(df['date'] >= oos_start) & (df['date'] <= oos_end)].copy()
        
        print("="*60)
        print("DATA SPLIT SUMMARY")
        print("="*60)
        print(f"Training set: {len(df_train):,} observations")
        print(f"  Date range: {df_train['date'].min()} to {df_train['date'].max()}")
        print(f"\nTest set: {len(df_test):,} observations")
        print(f"  Date range: {df_test['date'].min()} to {df_test['date'].max()}")
        print(f"\nOut-of-sample set: {len(df_oos):,} observations")
        print(f"  Date range: {df_oos['date'].min()} to {df_oos['date'].max()}")
        print("="*60)
        
        return df_train, df_test, df_oos
    
    def prepare_features_target(self, df):
        """Prepare feature matrix X and target vector y
        
        Args:
            df: DataFrame with features and target
            
        Returns:
            tuple: (X, y) where X is feature matrix and y is target vector
        """
        feature_cols = [col for col in df.columns 
                       if col not in ['date', 'spy_return', 'spy_return_next']]
        X = df[feature_cols].values
        y = df['spy_return_next'].values
        
        return X, y, feature_cols
    
    def train_elastic_net(self, X_train, y_train, cv_folds=5):
        """Train Elastic Net model with cross-validation
        
        Args:
            X_train: Training feature matrix
            y_train: Training target vector
            cv_folds: Number of CV folds
            
        Returns:
            dict: Best model and CV results
        """
        print("\nTraining Elastic Net model...")
        
        # Scale features for Elastic Net
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Hyperparameter grid
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        cv_scores = []
        best_score = -np.inf
        best_params = None
        
        for alpha in alphas:
            for l1_ratio in l1_ratios:
                fold_scores = []
                for train_idx, val_idx in tscv.split(X_train_scaled):
                    X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                    
                    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=42)
                    model.fit(X_fold_train, y_fold_train)
                    
                    y_pred = model.predict(X_fold_val)
                    score = r2_score(y_fold_val, y_pred)
                    fold_scores.append(score)
                
                avg_score = np.mean(fold_scores)
                cv_scores.append({
                    'alpha': alpha,
                    'l1_ratio': l1_ratio,
                    'mean_r2': avg_score,
                    'std_r2': np.std(fold_scores)
                })
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = {'alpha': alpha, 'l1_ratio': l1_ratio}
        
        # Train final model with best parameters
        best_model = ElasticNet(alpha=best_params['alpha'], 
                               l1_ratio=best_params['l1_ratio'],
                               max_iter=10000, 
                               random_state=42)
        best_model.fit(X_train_scaled, y_train)
        
        print(f"Best parameters: alpha={best_params['alpha']}, l1_ratio={best_params['l1_ratio']}")
        print(f"Best CV R²: {best_score:.4f}")
        
        return {
            'model': best_model,
            'scaler': scaler,
            'params': best_params,
            'cv_scores': pd.DataFrame(cv_scores)
        }
    
    def train_random_forest(self, X_train, y_train, cv_folds=5):
        """Train Random Forest model with cross-validation
        
        Args:
            X_train: Training feature matrix
            y_train: Training target vector
            cv_folds: Number of CV folds
            
        Returns:
            dict: Best model and CV results
        """
        print("\nTraining Random Forest model...")
        
        # Hyperparameter grid
        n_estimators_list = [50, 100, 200]
        max_depth_list = [5, 10, 20, None]
        min_samples_split_list = [2, 5, 10]
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        cv_scores = []
        best_score = -np.inf
        best_params = None
        
        for n_estimators in n_estimators_list:
            for max_depth in max_depth_list:
                for min_samples_split in min_samples_split_list:
                    fold_scores = []
                    for train_idx, val_idx in tscv.split(X_train):
                        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                        
                        model = RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            random_state=42,
                            n_jobs=-1
                        )
                        model.fit(X_fold_train, y_fold_train)
                        
                        y_pred = model.predict(X_fold_val)
                        score = r2_score(y_fold_val, y_pred)
                        fold_scores.append(score)
                    
                    avg_score = np.mean(fold_scores)
                    cv_scores.append({
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'mean_r2': avg_score,
                        'std_r2': np.std(fold_scores)
                    })
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_params = {
                            'n_estimators': n_estimators,
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split
                        }
        
        # Train final model with best parameters
        best_model = RandomForestRegressor(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            random_state=42,
            n_jobs=-1
        )
        best_model.fit(X_train, y_train)
        
        print(f"Best parameters: {best_params}")
        print(f"Best CV R²: {best_score:.4f}")
        
        return {
            'model': best_model,
            'scaler': None,  # RF doesn't need scaling
            'params': best_params,
            'cv_scores': pd.DataFrame(cv_scores)
        }
    
    def train_xgboost(self, X_train, y_train, cv_folds=5):
        """Train XGBoost model with cross-validation
        
        Args:
            X_train: Training feature matrix
            y_train: Training target vector
            cv_folds: Number of CV folds
            
        Returns:
            dict: Best model and CV results
        """
        print("\nTraining XGBoost model...")
        
        # Hyperparameter grid
        n_estimators_list = [50, 100, 200]
        max_depth_list = [3, 5, 7]
        learning_rate_list = [0.01, 0.1, 0.3]
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        cv_scores = []
        best_score = -np.inf
        best_params = None
        
        for n_estimators in n_estimators_list:
            for max_depth in max_depth_list:
                for learning_rate in learning_rate_list:
                    fold_scores = []
                    for train_idx, val_idx in tscv.split(X_train):
                        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                        
                        model = xgb.XGBRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            random_state=42,
                            n_jobs=-1
                        )
                        model.fit(X_fold_train, y_fold_train)
                        
                        y_pred = model.predict(X_fold_val)
                        score = r2_score(y_fold_val, y_pred)
                        fold_scores.append(score)
                    
                    avg_score = np.mean(fold_scores)
                    cv_scores.append({
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'learning_rate': learning_rate,
                        'mean_r2': avg_score,
                        'std_r2': np.std(fold_scores)
                    })
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_params = {
                            'n_estimators': n_estimators,
                            'max_depth': max_depth,
                            'learning_rate': learning_rate
                        }
        
        # Train final model with best parameters
        best_model = xgb.XGBRegressor(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            learning_rate=best_params['learning_rate'],
            random_state=42,
            n_jobs=-1
        )
        best_model.fit(X_train, y_train)
        
        print(f"Best parameters: {best_params}")
        print(f"Best CV R²: {best_score:.4f}")
        
        return {
            'model': best_model,
            'scaler': None,  # XGBoost doesn't need scaling
            'params': best_params,
            'cv_scores': pd.DataFrame(cv_scores)
        }
    
    def evaluate_model(self, model, scaler, X, y, set_name=''):
        """Evaluate model performance
        
        Args:
            model: Trained model
            scaler: Feature scaler (None if not needed)
            X: Feature matrix
            y: Target vector
            set_name: Name of the dataset (for printing)
            
        Returns:
            dict: Evaluation metrics
        """
        # Scale features if scaler is provided
        if scaler is not None:
            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled)
        else:
            y_pred = model.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Calculate directional accuracy
        direction_actual = np.sign(y)
        direction_pred = np.sign(y_pred)
        directional_accuracy = np.mean(direction_actual == direction_pred)
        
        results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': directional_accuracy,
            'mean_actual': np.mean(y),
            'mean_predicted': np.mean(y_pred),
            'std_actual': np.std(y),
            'std_predicted': np.std(y_pred)
        }
        
        if set_name:
            print(f"\n{set_name} Results:")
            print(f"  R²: {r2:.4f}")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  MAE: {mae:.6f}")
            print(f"  Directional Accuracy: {directional_accuracy:.4f}")
        
        return results
    
    def train_all_models(self, df_train, cv_folds=5):
        """Train all models with cross-validation
        
        Args:
            df_train: Training dataframe
            cv_folds: Number of CV folds
        """
        print("\n" + "="*60)
        print("TRAINING MODELS WITH TIME SERIES CROSS-VALIDATION")
        print("="*60)
        
        X_train, y_train, feature_cols = self.prepare_features_target(df_train)
        print(f"\nTraining on {len(X_train):,} samples with {len(feature_cols)} features")
        
        # Train Elastic Net
        self.models['elastic_net'] = self.train_elastic_net(X_train, y_train, cv_folds)
        
        # Train Random Forest
        self.models['random_forest'] = self.train_random_forest(X_train, y_train, cv_folds)
        
        # Train XGBoost
        self.models['xgboost'] = self.train_xgboost(X_train, y_train, cv_folds)
        
        print("\n" + "="*60)
        print("MODEL TRAINING COMPLETE")
        print("="*60)
    
    def evaluate_all_models(self, df_test, df_oos):
        """Evaluate all models on test and out-of-sample sets
        
        Args:
            df_test: Test dataframe
            df_oos: Out-of-sample dataframe
        """
        print("\n" + "="*60)
        print("EVALUATING MODELS")
        print("="*60)
        
        X_test, y_test, _ = self.prepare_features_target(df_test)
        X_oos, y_oos, _ = self.prepare_features_target(df_oos)
        
        self.test_results = {}
        self.oos_results = {}
        
        for model_name, model_dict in self.models.items():
            print(f"\n{model_name.upper().replace('_', ' ')}:")
            print("-" * 40)
            
            # Evaluate on test set
            test_results = self.evaluate_model(
                model_dict['model'],
                model_dict['scaler'],
                X_test,
                y_test,
                set_name='Test Set'
            )
            self.test_results[model_name] = test_results
            
            # Evaluate on out-of-sample set
            oos_results = self.evaluate_model(
                model_dict['model'],
                model_dict['scaler'],
                X_oos,
                y_oos,
                set_name='Out-of-Sample Set'
            )
            self.oos_results[model_name] = oos_results
    
    def get_results_summary(self):
        """Get summary of all results
        
        Returns:
            DataFrame: Summary of results across all models and datasets
        """
        results_list = []
        
        for model_name in self.models.keys():
            # CV results
            cv_r2 = self.models[model_name]['cv_scores']['mean_r2'].max()
            cv_std = self.models[model_name]['cv_scores'].loc[
                self.models[model_name]['cv_scores']['mean_r2'].idxmax(), 'std_r2'
            ]
            
            # Test results
            test_r2 = self.test_results[model_name]['r2']
            test_rmse = self.test_results[model_name]['rmse']
            test_dir_acc = self.test_results[model_name]['directional_accuracy']
            
            # OOS results
            oos_r2 = self.oos_results[model_name]['r2']
            oos_rmse = self.oos_results[model_name]['rmse']
            oos_dir_acc = self.oos_results[model_name]['directional_accuracy']
            
            results_list.append({
                'model': model_name,
                'cv_r2_mean': cv_r2,
                'cv_r2_std': cv_std,
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'test_dir_acc': test_dir_acc,
                'oos_r2': oos_r2,
                'oos_rmse': oos_rmse,
                'oos_dir_acc': oos_dir_acc
            })
        
        return pd.DataFrame(results_list)
    
    def run_full_pipeline(self, df, cv_folds=5):
        """Run the complete modeling pipeline
        
        Args:
            df: Full dataframe with features
            cv_folds: Number of CV folds (default: 5)
            
        Returns:
            DataFrame: Results summary
        """
        # Split data
        df_train, df_test, df_oos = self.split_data_by_date(df)
        
        # Train models
        self.train_all_models(df_train, cv_folds)
        
        # Evaluate models
        self.evaluate_all_models(df_test, df_oos)
        
        # Get summary
        results_summary = self.get_results_summary()
        
        return results_summary


def main():
    """Main function to run the modeling pipeline"""
    from data_loader import DataLoader
    from feature_extractor import ArticleFeatureExtractor
    from feature_analyzer import FeatureAnalyzer
    
    print("="*60)
    print("NEWS SENTIMENT MODELING PIPELINE")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    data_loader = DataLoader()
    df_news = data_loader.load_news_dataset()
    df_spy = data_loader.load_spy_returns()
    
    # Extract features
    print("\n2. Extracting features...")
    extractor = ArticleFeatureExtractor(data_loader=data_loader)
    df_features = extractor.compute_all_features(df_news, reload_cache=True)
    
    # Prepare features for modeling
    print("\n3. Preparing features for modeling...")
    feature_analyzer = FeatureAnalyzer()
    df_clean = feature_analyzer.prepare_features_for_modeling(df_features, df_spy)
    
    # Run modeling pipeline
    print("\n4. Running modeling pipeline...")
    modeler = NewsSentimentModeler()
    results_summary = modeler.run_full_pipeline(df_clean, cv_folds=5)
    
    # Display results
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(results_summary.to_string(index=False))
    
    return modeler, results_summary


# if __name__ == "__main__":
#     modeler, results = main()

