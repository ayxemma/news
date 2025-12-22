"""
Modeling script for SPY returns prediction using news features
Uses time series cross-validation and tests multiple models
"""
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
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
        """Train Elastic Net model with GridSearchCV
        
        Args:
            X_train: Training feature matrix
            y_train: Training target vector
            cv_folds: Number of CV folds
            
        Returns:
            dict: Best model and CV results
        """
        print("\nTraining Elastic Net model...")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Create pipeline with scaler and model
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(max_iter=20000, random_state=42))
        ])
        
        # Hyperparameter grid
        param_grid = {
            "model__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
            "model__l1_ratio": [0.0, 0.2, 0.5, 0.8, 1.0]
        }
        
        # Create R² scorer (GridSearchCV maximizes, so we use R² directly)
        r2_scorer = make_scorer(r2_score)
        
        # GridSearchCV
        gscv = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=tscv,
            scoring=r2_scorer,
            n_jobs=-1,
            return_train_score=True
        )
        
        gscv.fit(X_train, y_train)
        
        # Extract best model and scaler from pipeline
        best_model = gscv.best_estimator_.named_steps['model']
        scaler = gscv.best_estimator_.named_steps['scaler']
        
        # Extract best parameters (remove 'model__' prefix)
        best_params = {k.replace('model__', ''): v for k, v in gscv.best_params_.items()}
        
        # Create CV scores DataFrame from GridSearchCV results
        cv_results = gscv.cv_results_
        cv_scores = []
        for i, params in enumerate(cv_results['params']):
            cv_scores.append({
                'alpha': params['model__alpha'],
                'l1_ratio': params['model__l1_ratio'],
                'mean_r2': cv_results['mean_test_score'][i],
                'std_r2': cv_results['std_test_score'][i]
            })
        
        print(f"Best parameters: alpha={best_params['alpha']}, l1_ratio={best_params['l1_ratio']}")
        print(f"Best CV R²: {gscv.best_score_:.4f}")
        
        return {
            'model': best_model,
            'scaler': scaler,
            'params': best_params,
            'cv_scores': pd.DataFrame(cv_scores),
            'grid_search': gscv
        }
    
    def train_random_forest(self, X_train, y_train, cv_folds=5):
        """Train Random Forest model with GridSearchCV
        
        Args:
            X_train: Training feature matrix
            y_train: Training target vector
            cv_folds: Number of CV folds
            
        Returns:
            dict: Best model and CV results
        """
        print("\nTraining Random Forest model...")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Base model
        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [200, 500],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10]
        }
        
        # Create R² scorer
        r2_scorer = make_scorer(r2_score)
        
        # GridSearchCV
        gscv = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=tscv,
            scoring=r2_scorer,
            n_jobs=-1,
            return_train_score=True
        )
        
        gscv.fit(X_train, y_train)
        
        # Extract best model
        best_model = gscv.best_estimator_
        best_params = gscv.best_params_
        
        # Create CV scores DataFrame from GridSearchCV results
        cv_results = gscv.cv_results_
        cv_scores = []
        for i, params in enumerate(cv_results['params']):
            cv_scores.append({
                'n_estimators': params['n_estimators'],
                'max_depth': params['max_depth'],
                'min_samples_split': params['min_samples_split'],
                'mean_r2': cv_results['mean_test_score'][i],
                'std_r2': cv_results['std_test_score'][i]
            })
        
        print(f"Best parameters: {best_params}")
        print(f"Best CV R²: {gscv.best_score_:.4f}")
        
        return {
            'model': best_model,
            'scaler': None,  # RF doesn't need scaling
            'params': best_params,
            'cv_scores': pd.DataFrame(cv_scores),
            'grid_search': gscv
        }
    
    def train_xgboost(self, X_train, y_train, cv_folds=5):
        """Train XGBoost model with GridSearchCV
        
        Args:
            X_train: Training feature matrix
            y_train: Training target vector
            cv_folds: Number of CV folds
            
        Returns:
            dict: Best model and CV results
        """
        print("\nTraining XGBoost model...")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Base model
        base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [200, 500],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3]
        }
        
        # Create R² scorer
        r2_scorer = make_scorer(r2_score)
        
        # GridSearchCV
        gscv = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=tscv,
            scoring=r2_scorer,
            n_jobs=-1,
            return_train_score=True
        )
        
        gscv.fit(X_train, y_train)
        
        # Extract best model
        best_model = gscv.best_estimator_
        best_params = gscv.best_params_
        
        # Create CV scores DataFrame from GridSearchCV results
        cv_results = gscv.cv_results_
        cv_scores = []
        for i, params in enumerate(cv_results['params']):
            cv_scores.append({
                'n_estimators': params['n_estimators'],
                'max_depth': params['max_depth'],
                'learning_rate': params['learning_rate'],
                'mean_r2': cv_results['mean_test_score'][i],
                'std_r2': cv_results['std_test_score'][i]
            })
        
        print(f"Best parameters: {best_params}")
        print(f"Best CV R²: {gscv.best_score_:.4f}")
        
        return {
            'model': best_model,
            'scaler': None,  # XGBoost doesn't need scaling
            'params': best_params,
            'cv_scores': pd.DataFrame(cv_scores),
            'grid_search': gscv
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
    
    def predict_ensemble(self, X):
        """Make ensemble predictions by averaging predictions from all models
        
        Args:
            X: Feature matrix
            
        Returns:
            numpy array: Ensemble predictions (simple average of all model predictions)
        """
        if not self.models:
            raise ValueError("No models trained yet. Call train_all_models() first.")
        
        predictions_list = []
        
        for model_name, model_dict in self.models.items():
            if model_dict['scaler'] is not None:
                X_scaled = model_dict['scaler'].transform(X)
                y_pred = model_dict['model'].predict(X_scaled)
            else:
                y_pred = model_dict['model'].predict(X)
            predictions_list.append(y_pred)
        
        # Simple average of all predictions
        ensemble_pred = np.mean(predictions_list, axis=0)
        
        return ensemble_pred
    
    def evaluate_ensemble(self, X, y, set_name=''):
        """Evaluate ensemble model performance
        
        Args:
            X: Feature matrix
            y: Target vector
            set_name: Name of the dataset (for printing)
            
        Returns:
            dict: Evaluation metrics for ensemble model
        """
        y_pred = self.predict_ensemble(X)
        
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
            print(f"\n{set_name} Results (Ensemble):")
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
        
        # Evaluate ensemble model
        print("\n" + "="*60)
        print("ENSEMBLE MODEL (Simple Average)")
        print("="*60)
        
        # Evaluate ensemble model
        if len(self.models) > 0:
            print("\n" + "="*60)
            print("ENSEMBLE MODEL (Simple Average)")
            print("="*60)
            
            # Evaluate ensemble on test set
            ensemble_test_results = self.evaluate_ensemble(X_test, y_test, set_name='Test Set')
            self.test_results['ensemble'] = ensemble_test_results
            
            # Evaluate ensemble on out-of-sample set
            ensemble_oos_results = self.evaluate_ensemble(X_oos, y_oos, set_name='Out-of-Sample Set')
            self.oos_results['ensemble'] = ensemble_oos_results
    
    def get_results_summary(self):
        """Get summary of all results including ensemble
        
        Returns:
            DataFrame: Summary of results across all models and datasets
        """
        results_list = []
        
        # Add individual models
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
        
        # Add ensemble model (if evaluated)
        if 'ensemble' in self.test_results and 'ensemble' in self.oos_results:
            results_list.append({
                'model': 'ensemble',
                'cv_r2_mean': np.nan,  # Ensemble doesn't have CV score
                'cv_r2_std': np.nan,
                'test_r2': self.test_results['ensemble']['r2'],
                'test_rmse': self.test_results['ensemble']['rmse'],
                'test_dir_acc': self.test_results['ensemble']['directional_accuracy'],
                'oos_r2': self.oos_results['ensemble']['r2'],
                'oos_rmse': self.oos_results['ensemble']['rmse'],
                'oos_dir_acc': self.oos_results['ensemble']['directional_accuracy']
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
    
    def save_best_hyperparameters(self, filepath='model_hyperparameters.json'):
        """Save best hyperparameters from time series cross-validation
        
        Saves the best hyperparameters for each model (Elastic Net, Random Forest, XGBoost)
        along with their CV performance metrics to a JSON file.
        
        Args:
            filepath: Path to save the hyperparameters JSON file (default: 'model_hyperparameters.json')
            
        Returns:
            dict: Dictionary containing saved hyperparameters and metrics
            
        Example:
            >>> modeler = NewsSentimentModeler()
            >>> modeler.train_all_models(df_train)
            >>> modeler.save_best_hyperparameters('best_params.json')
        """
        if not self.models:
            raise ValueError("No models trained yet. Call train_all_models() first.")
        
        hyperparameters = {}
        
        for model_name, model_dict in self.models.items():
            # Get best parameters
            best_params = model_dict['params']
            
            # Get CV performance metrics
            cv_scores_df = model_dict['cv_scores']
            best_cv_score = cv_scores_df['mean_r2'].max()
            best_cv_std = cv_scores_df.loc[
                cv_scores_df['mean_r2'].idxmax(), 'std_r2'
            ]
            
            hyperparameters[model_name] = {
                'hyperparameters': best_params,
                'cv_performance': {
                    'mean_r2': float(best_cv_score),
                    'std_r2': float(best_cv_std)
                }
            }
            
            # Add test and OOS results if available
            if model_name in self.test_results:
                hyperparameters[model_name]['test_performance'] = {
                    'r2': float(self.test_results[model_name]['r2']),
                    'rmse': float(self.test_results[model_name]['rmse']),
                    'mae': float(self.test_results[model_name]['mae']),
                    'directional_accuracy': float(self.test_results[model_name]['directional_accuracy'])
                }
            
            if model_name in self.oos_results:
                hyperparameters[model_name]['oos_performance'] = {
                    'r2': float(self.oos_results[model_name]['r2']),
                    'rmse': float(self.oos_results[model_name]['rmse']),
                    'mae': float(self.oos_results[model_name]['mae']),
                    'directional_accuracy': float(self.oos_results[model_name]['directional_accuracy'])
                }
        
        # Save to JSON file
        with open(filepath, 'w') as f:
            json.dump(hyperparameters, f, indent=4)
        
        print(f"\nHyperparameters saved to {filepath}")
        print("="*60)
        print("SAVED HYPERPARAMETERS SUMMARY")
        print("="*60)
        for model_name, params_dict in hyperparameters.items():
            print(f"\n{model_name.upper().replace('_', ' ')}:")
            print(f"  Hyperparameters: {params_dict['hyperparameters']}")
            print(f"  CV R²: {params_dict['cv_performance']['mean_r2']:.4f} "
                  f"(±{params_dict['cv_performance']['std_r2']:.4f})")
            if 'test_performance' in params_dict:
                print(f"  Test R²: {params_dict['test_performance']['r2']:.4f}")
            if 'oos_performance' in params_dict:
                print(f"  OOS R²: {params_dict['oos_performance']['r2']:.4f}")
        
        return hyperparameters
    
    def load_best_hyperparameters(self, filepath='model_hyperparameters.json'):
        """Load best hyperparameters from saved training results
        
        Loads previously saved hyperparameters and performance metrics from a JSON file.
        This allows you to reuse the best hyperparameters found during cross-validation
        without retraining models.
        
        Args:
            filepath: Path to the hyperparameters JSON file (default: 'model_hyperparameters.json')
            
        Returns:
            dict: Dictionary containing loaded hyperparameters and metrics
            
        Example:
            >>> modeler = NewsSentimentModeler()
            >>> hyperparams = modeler.load_best_hyperparameters('best_params.json')
            >>> # Use hyperparams to recreate models with best parameters
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Hyperparameters file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            hyperparameters = json.load(f)
        
        print(f"\nHyperparameters loaded from {filepath}")
        print("="*60)
        print("LOADED HYPERPARAMETERS SUMMARY")
        print("="*60)
        for model_name, params_dict in hyperparameters.items():
            print(f"\n{model_name.upper().replace('_', ' ')}:")
            print(f"  Hyperparameters: {params_dict['hyperparameters']}")
            print(f"  CV R²: {params_dict['cv_performance']['mean_r2']:.4f} "
                  f"(±{params_dict['cv_performance']['std_r2']:.4f})")
            if 'test_performance' in params_dict:
                print(f"  Test R²: {params_dict['test_performance']['r2']:.4f}")
            if 'oos_performance' in params_dict:
                print(f"  OOS R²: {params_dict['oos_performance']['r2']:.4f}")
        
        return hyperparameters
    
    def recreate_models_from_hyperparameters(self, hyperparameters, X_train, y_train):
        """Recreate models using loaded hyperparameters
        
        Trains new models using the hyperparameters loaded from a saved file.
        This is useful when you want to retrain models with the same hyperparameters
        on new data or recreate models without running cross-validation again.
        
        Args:
            hyperparameters: Dictionary of hyperparameters (from load_best_hyperparameters)
            X_train: Training feature matrix
            y_train: Training target vector
            
        Example:
            >>> modeler = NewsSentimentModeler()
            >>> hyperparams = modeler.load_best_hyperparameters('best_params.json')
            >>> modeler.recreate_models_from_hyperparameters(hyperparams, X_train, y_train)
        """
        print("\n" + "="*60)
        print("RECREATING MODELS FROM SAVED HYPERPARAMETERS")
        print("="*60)
        
        self.models = {}
        
        # Recreate Elastic Net
        if 'elastic_net' in hyperparameters:
            print("\nRecreating Elastic Net model...")
            params = hyperparameters['elastic_net']['hyperparameters']
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            model = ElasticNet(
                alpha=params['alpha'],
                l1_ratio=params['l1_ratio'],
                max_iter=20000,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
            
            self.models['elastic_net'] = {
                'model': model,
                'scaler': scaler,
                'params': params,
                'cv_scores': pd.DataFrame([hyperparameters['elastic_net']['cv_performance']])
            }
            print(f"  Trained with alpha={params['alpha']}, l1_ratio={params['l1_ratio']}")
        
        # Recreate Random Forest
        if 'random_forest' in hyperparameters:
            print("\nRecreating Random Forest model...")
            params = hyperparameters['random_forest']['hyperparameters']
            
            model = RandomForestRegressor(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            self.models['random_forest'] = {
                'model': model,
                'scaler': None,
                'params': params,
                'cv_scores': pd.DataFrame([hyperparameters['random_forest']['cv_performance']])
            }
            print(f"  Trained with {params}")
        
        # Recreate XGBoost
        if 'xgboost' in hyperparameters:
            print("\nRecreating XGBoost model...")
            params = hyperparameters['xgboost']['hyperparameters']
            
            model = xgb.XGBRegressor(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            self.models['xgboost'] = {
                'model': model,
                'scaler': None,
                'params': params,
                'cv_scores': pd.DataFrame([hyperparameters['xgboost']['cv_performance']])
            }
            print(f"  Trained with {params}")
        
        print("\n" + "="*60)
        print("MODELS RECREATED SUCCESSFULLY")
        print("="*60)
    
    def load_and_predict(self, hyperparameters_file, X_test, y_test=None):
        """Load best hyperparameters and make predictions on test data
        
        Loads saved hyperparameters, recreates models, and makes predictions.
        Optionally evaluates performance if y_test is provided.
        
        Args:
            hyperparameters_file: Path to JSON file with saved hyperparameters
            X_test: Test feature matrix
            y_test: Optional test target vector for evaluation
            
        Returns:
            dict: Predictions and optionally evaluation metrics for each model
            
        Example:
            >>> modeler = NewsSentimentModeler()
            >>> X_test, y_test, _ = modeler.prepare_features_target(df_test)
            >>> results = modeler.load_and_predict('best_params.json', X_test, y_test)
        """
        # Load hyperparameters
        hyperparameters = self.load_best_hyperparameters(hyperparameters_file)
        
        # Recreate models (need X_train for fitting, but we'll use a dummy for now)
        # Actually, we need to fit models, so we need training data
        print("\nWarning: Models need to be trained before prediction.")
        print("Please use recreate_models_from_hyperparameters() first with training data.")
        
        predictions = {}
        evaluations = {}
        
        if self.models:
            for model_name, model_dict in self.models.items():
                if model_name in hyperparameters:
                    # Make predictions
                    if model_dict['scaler'] is not None:
                        X_test_scaled = model_dict['scaler'].transform(X_test)
                        y_pred = model_dict['model'].predict(X_test_scaled)
                    else:
                        y_pred = model_dict['model'].predict(X_test)
                    
                    predictions[model_name] = y_pred
                    
                    # Evaluate if y_test provided
                    if y_test is not None:
                        eval_results = self.evaluate_model(
                            model_dict['model'],
                            model_dict['scaler'],
                            X_test,
                            y_test,
                            set_name=f'{model_name} (Test)'
                        )
                        evaluations[model_name] = eval_results
        
        return {
            'predictions': predictions,
            'evaluations': evaluations if y_test is not None else None
        }
    
    def visualize_model_performance(self, figsize=(14, 8)):
        """Visualize predictive performance across models using R² and RMSE
        
        Creates comparison plots showing model performance metrics across
        cross-validation, test, and out-of-sample datasets. Includes ensemble model.
        
        Args:
            figsize: Figure size tuple (default: (14, 8))
            
        Example:
            >>> modeler = NewsSentimentModeler()
            >>> modeler.train_all_models(df_train)
            >>> modeler.evaluate_all_models(df_test, df_oos)
            >>> modeler.visualize_model_performance()
        """
        if not self.models or not self.test_results or not self.oos_results:
            print("Error: Models must be trained and evaluated first.")
            print("Call train_all_models() and evaluate_all_models() before visualization.")
            return
        
        # Prepare data for visualization - include ensemble if available
        model_names = list(self.models.keys())
        if 'ensemble' in self.test_results:
            model_names.append('ensemble')
        
        model_names_display = [name.replace('_', ' ').title() for name in model_names]
        
        # Extract metrics
        cv_r2 = []
        test_r2 = []
        oos_r2 = []
        test_rmse = []
        oos_rmse = []
        
        for model_name in model_names:
            # CV R² (ensemble doesn't have CV score)
            if model_name == 'ensemble':
                cv_r2.append(np.nan)
            else:
                cv_r2_val = self.models[model_name]['cv_scores']['mean_r2'].max()
                cv_r2.append(cv_r2_val)
            
            # Test metrics
            test_r2.append(self.test_results[model_name]['r2'])
            test_rmse.append(self.test_results[model_name]['rmse'])
            
            # OOS metrics
            oos_r2.append(self.oos_results[model_name]['r2'])
            oos_rmse.append(self.oos_results[model_name]['rmse'])
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        x_pos = np.arange(len(model_names))
        width = 0.25
        
        # Plot 1: R² Comparison across datasets
        # Handle NaN values for ensemble CV score
        cv_r2_plot = [val if not np.isnan(val) else 0 for val in cv_r2]
        cv_mask = [not np.isnan(val) for val in cv_r2]
        
        axes[0, 0].bar(x_pos - width, cv_r2_plot, width, label='CV', alpha=0.8, color='skyblue')
        axes[0, 0].bar(x_pos, test_r2, width, label='Test', alpha=0.8, color='lightgreen')
        axes[0, 0].bar(x_pos + width, oos_r2, width, label='Out-of-Sample', alpha=0.8, color='coral')
        axes[0, 0].set_xlabel('Model', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('R² Score', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('R² Score Comparison', fontsize=13, fontweight='bold')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(model_names_display, rotation=15, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add value labels (skip CV label for ensemble)
        for i, (cv, test, oos, has_cv) in enumerate(zip(cv_r2, test_r2, oos_r2, cv_mask)):
            if has_cv:
                axes[0, 0].text(i - width, cv, f'{cv:.3f}', ha='center', va='bottom', fontsize=8)
            axes[0, 0].text(i, test, f'{test:.3f}', ha='center', va='bottom', fontsize=8)
            axes[0, 0].text(i + width, oos, f'{oos:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: RMSE Comparison
        axes[0, 1].bar(x_pos - width/2, test_rmse, width, label='Test', alpha=0.8, color='lightgreen')
        axes[0, 1].bar(x_pos + width/2, oos_rmse, width, label='Out-of-Sample', alpha=0.8, color='coral')
        axes[0, 1].set_xlabel('Model', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('RMSE', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('RMSE Comparison', fontsize=13, fontweight='bold')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(model_names_display, rotation=15, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (test, oos) in enumerate(zip(test_rmse, oos_rmse)):
            axes[0, 1].text(i - width/2, test, f'{test:.4f}', ha='center', va='bottom', fontsize=8)
            axes[0, 1].text(i + width/2, oos, f'{oos:.4f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 3: Performance Summary Table
        axes[1, 0].axis('off')
        summary_data = []
        for i, model_name in enumerate(model_names):
            cv_str = f'{cv_r2[i]:.4f}' if not np.isnan(cv_r2[i]) else 'N/A'
            summary_data.append([
                model_names_display[i],
                cv_str,
                f'{test_r2[i]:.4f}',
                f'{oos_r2[i]:.4f}',
                f'{test_rmse[i]:.6f}',
                f'{oos_rmse[i]:.6f}'
            ])
        
        table = axes[1, 0].table(
            cellText=summary_data,
            colLabels=['Model', 'CV R²', 'Test R²', 'OOS R²', 'Test RMSE', 'OOS RMSE'],
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(6):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        axes[1, 0].set_title('Performance Summary', fontsize=13, fontweight='bold', pad=20)
        
        # Plot 4: R² vs RMSE Scatter (Test Set)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Added color for ensemble
        for i, model_name in enumerate(model_names):
            marker = 's' if model_name == 'ensemble' else 'o'  # Square for ensemble
            axes[1, 1].scatter(test_rmse[i], test_r2[i], s=200, alpha=0.7, 
                              color=colors[i % len(colors)], label=model_names_display[i],
                              marker=marker, edgecolors='black', linewidths=1.5 if model_name == 'ensemble' else 0.5)
            axes[1, 1].annotate(model_names_display[i], 
                               (test_rmse[i], test_r2[i]),
                               xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        axes[1, 1].set_xlabel('RMSE (Test)', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('R² Score (Test)', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('R² vs RMSE (Test Set)', fontsize=13, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
        
        return fig, axes
    
    def load_and_evaluate_on_test(self, hyperparameters_file, df_train, df_test):
        """Load best hyperparameters, recreate models, and evaluate on test data
        
        Complete workflow: loads hyperparameters, recreates models on training data,
        and evaluates on test data with visualization.
        
        Args:
            hyperparameters_file: Path to JSON file with saved hyperparameters
            df_train: Training dataframe
            df_test: Test dataframe
            
        Returns:
            dict: Evaluation results for all models
            
        Example:
            >>> modeler = NewsSentimentModeler()
            >>> results = modeler.load_and_evaluate_on_test('best_params.json', df_train, df_test)
        """
        print("="*60)
        print("LOADING MODELS AND EVALUATING ON TEST DATA")
        print("="*60)
        
        # Load hyperparameters
        hyperparameters = self.load_best_hyperparameters(hyperparameters_file)
        
        # Prepare training and test data
        X_train, y_train, _ = self.prepare_features_target(df_train)
        X_test, y_test, _ = self.prepare_features_target(df_test)
        
        # Recreate models from hyperparameters
        self.recreate_models_from_hyperparameters(hyperparameters, X_train, y_train)
        
        # Evaluate on test set
        print("\n" + "="*60)
        print("EVALUATING ON TEST DATA")
        print("="*60)
        
        test_results = {}
        for model_name, model_dict in self.models.items():
            print(f"\n{model_name.upper().replace('_', ' ')}:")
            print("-" * 40)
            
            eval_results = self.evaluate_model(
                model_dict['model'],
                model_dict['scaler'],
                X_test,
                y_test,
                set_name='Test Set'
            )
            test_results[model_name] = eval_results
        
        # Evaluate ensemble model
        print("\n" + "="*60)
        print("ENSEMBLE MODEL (Simple Average)")
        print("="*60)
        ensemble_test_results = self.evaluate_ensemble(X_test, y_test, set_name='Test Set')
        test_results['ensemble'] = ensemble_test_results
        
        # Store test results
        self.test_results = test_results
        
        # Visualize performance
        print("\n" + "="*60)
        print("VISUALIZING MODEL PERFORMANCE")
        print("="*60)
        self.visualize_model_performance()
        
        return test_results


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

