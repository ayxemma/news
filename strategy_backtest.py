"""
Strategy Backtesting Module
Implements long-short portfolio construction and backtesting for SPY predictions
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
from scipy import stats
warnings.filterwarnings('ignore')


class StrategyBacktester:
    """Class for constructing and backtesting long-short strategies based on model predictions"""
    
    def __init__(self):
        """Initialize the backtester"""
        self.strategies = {}
        self.results = {}
    
    def normalize_signal_zscore(self, signal, window=60, min_periods=30):
        """Normalize prediction signal using rolling z-score
        
        Args:
            signal: Series or array of model predictions
            window: Rolling window size in days (default: 60)
            min_periods: Minimum periods for rolling calculation (default: 30)
            
        Returns:
            Series: Z-score normalized signal
        """
        signal_series = pd.Series(signal) if not isinstance(signal, pd.Series) else signal
        
        rolling_mean = signal_series.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = signal_series.rolling(window=window, min_periods=min_periods).std()
        
        z_score = (signal_series - rolling_mean) / rolling_std
        z_score = z_score.fillna(0)  # Fill early periods with 0 (neutral)
        
        return z_score
    
    def normalize_signal_rank(self, signal, window=60, min_periods=30):
        """Normalize prediction signal using rolling rank percentile
        
        Args:
            signal: Series or array of model predictions
            window: Rolling window size in days (default: 60)
            min_periods: Minimum periods for rolling calculation (default: 30)
            
        Returns:
            Series: Rank percentile normalized signal (0 to 1, centered at 0.5)
        """
        signal_series = pd.Series(signal) if not isinstance(signal, pd.Series) else signal
        
        rank_pct = signal_series.rolling(window=window, min_periods=min_periods).apply(
            lambda x: stats.rankdata(x)[-1] / len(x) if len(x) > 0 else 0.5
        )
        
        # Center at 0 (convert from 0-1 to -0.5 to 0.5)
        rank_pct = (rank_pct - 0.5) * 2
        
        return rank_pct.fillna(0)
    
    def compute_positions_continuous(self, z_signal, k=0.5, w_max=1.0):
        """Compute continuous positions from normalized signal
        
        Args:
            z_signal: Z-score normalized signal
            k: Scaling factor (default: 0.5)
            w_max: Maximum absolute position (default: 1.0 = 100% long/short)
            
        Returns:
            Series: Position weights
        """
        positions = z_signal * k
        positions = np.clip(positions, -w_max, w_max)
        return pd.Series(positions, index=z_signal.index if isinstance(z_signal, pd.Series) else None)
    
    def compute_positions_discrete(self, z_signal, threshold=0.5):
        """Compute discrete positions from normalized signal
        
        Args:
            z_signal: Z-score normalized signal
            threshold: Z-score threshold for taking positions (default: 0.5)
            
        Returns:
            Series: Position weights (-1, 0, or +1)
        """
        positions = pd.Series(0, index=z_signal.index if isinstance(z_signal, pd.Series) else None)
        positions[z_signal > threshold] = 1
        positions[z_signal < -threshold] = -1
        
        return positions
    
    def apply_volatility_targeting(self, positions, spy_returns, target_vol=0.15, window=60):
        """Apply volatility targeting to positions
        
        Args:
            positions: Series of position weights
            spy_returns: Series of SPY returns
            target_vol: Target annualized volatility (default: 0.15 = 15%)
            window: Rolling window for volatility estimation (default: 60)
            
        Returns:
            Series: Volatility-scaled positions
        """
        # Estimate rolling volatility of SPY
        rolling_vol = spy_returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        
        # Scale positions inversely to volatility
        vol_scale = target_vol / rolling_vol
        vol_scale = vol_scale.fillna(1.0)  # Use unscaled if vol not available
        
        # Apply scaling
        scaled_positions = positions * vol_scale
        
        # Re-clip to max position
        scaled_positions = np.clip(scaled_positions, -1.0, 1.0)
        
        return scaled_positions
    
    def compute_portfolio_returns(self, positions, spy_returns):
        """Compute portfolio returns from positions
        
        Args:
            positions: Series of position weights
            spy_returns: Series of SPY returns
            
        Returns:
            DataFrame: Portfolio returns and related metrics
        """
        # Align indices
        aligned = pd.DataFrame({
            'position': positions,
            'spy_return': spy_returns
        }).dropna()
        
        # Compute portfolio returns
        aligned['return'] = aligned['position'] * aligned['spy_return']
        
        # Cumulative returns
        aligned['cumulative_return'] = (1 + aligned['return']).cumprod()
        
        return aligned
    
    def compute_performance_metrics(self, returns, periods_per_year=252):
        """Compute comprehensive performance metrics
        
        Args:
            returns: Series of portfolio returns
            periods_per_year: Trading periods per year (default: 252)
            
        Returns:
            dict: Performance metrics
        """
        returns_clean = returns.dropna()
        
        if len(returns_clean) == 0:
            return {
                'total_return': np.nan,
                'annualized_return': np.nan,
                'annualized_volatility': np.nan,
                'sharpe_ratio': np.nan,
                'max_drawdown': np.nan,
                'calmar_ratio': np.nan,
                'hit_rate': np.nan,
                'win_rate': np.nan,
                'avg_win': np.nan,
                'avg_loss': np.nan,
                'profit_factor': np.nan,
                'turnover': np.nan
            }
        
        # Basic metrics
        total_return = (1 + returns_clean).prod() - 1
        n_periods = len(returns_clean)
        annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
        annualized_vol = returns_clean.std() * np.sqrt(periods_per_year)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else np.nan
        
        # Drawdown analysis
        cumulative = (1 + returns_clean).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else np.nan
        
        # Hit rate and win/loss stats
        hit_rate = (returns_clean > 0).mean()
        wins = returns_clean[returns_clean > 0]
        losses = returns_clean[returns_clean < 0]
        win_rate = len(wins) / len(returns_clean) if len(returns_clean) > 0 else np.nan
        avg_win = wins.mean() if len(wins) > 0 else np.nan
        avg_loss = losses.mean() if len(losses) > 0 else np.nan
        profit_factor = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() < 0 else np.nan
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'hit_rate': hit_rate,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'turnover': np.nan  # Will be computed separately
        }
    
    def compute_turnover(self, positions):
        """Compute average daily turnover
        
        Args:
            positions: Series of position weights
            
        Returns:
            float: Average daily turnover
        """
        position_changes = positions.diff().abs()
        avg_turnover = position_changes.mean()
        return avg_turnover
    
    def backtest_strategy(self, predictions, spy_returns, dates=None,
                         normalization='zscore', window=60, k=0.5, w_max=1.0,
                         threshold=0.5, position_type='continuous',
                         vol_targeting=False, target_vol=0.15,
                         strategy_name='strategy'):
        """Complete backtest of a trading strategy
        
        Args:
            predictions: Series or array of model predictions
            spy_returns: Series of SPY returns (aligned with predictions)
            dates: Optional date index
            normalization: 'zscore' or 'rank' (default: 'zscore')
            window: Rolling window for normalization (default: 60)
            k: Scaling factor for continuous positions (default: 0.5)
            w_max: Maximum position size (default: 1.0)
            threshold: Threshold for discrete positions (default: 0.5)
            position_type: 'continuous' or 'discrete' (default: 'continuous')
            vol_targeting: Whether to apply volatility targeting (default: False)
            target_vol: Target annualized volatility (default: 0.15)
            strategy_name: Name for this strategy
            
        Returns:
            dict: Complete backtest results
        """
        # Convert to Series with proper index
        if isinstance(predictions, np.ndarray):
            if dates is not None:
                predictions = pd.Series(predictions, index=dates)
            else:
                predictions = pd.Series(predictions)
        
        if isinstance(spy_returns, np.ndarray):
            if dates is not None:
                spy_returns = pd.Series(spy_returns, index=dates)
            else:
                spy_returns = pd.Series(spy_returns)
        
        # Align predictions and returns
        aligned = pd.DataFrame({
            'prediction': predictions,
            'spy_return': spy_returns
        }).dropna()
        
        if len(aligned) == 0:
            raise ValueError("No overlapping data between predictions and SPY returns")
        
        # Step 1: Normalize signal
        if normalization == 'zscore':
            z_signal = self.normalize_signal_zscore(aligned['prediction'], window=window)
        elif normalization == 'rank':
            z_signal = self.normalize_signal_rank(aligned['prediction'], window=window)
        else:
            raise ValueError(f"Unknown normalization method: {normalization}")
        
        # Step 2: Compute positions
        if position_type == 'continuous':
            positions = self.compute_positions_continuous(z_signal, k=k, w_max=w_max)
        elif position_type == 'discrete':
            positions = self.compute_positions_discrete(z_signal, threshold=threshold)
        else:
            raise ValueError(f"Unknown position type: {position_type}")
        
        # Step 3: Apply volatility targeting if requested
        if vol_targeting:
            positions = self.apply_volatility_targeting(
                positions, aligned['spy_return'], target_vol=target_vol, window=window
            )
        
        # Step 4: Compute portfolio returns
        portfolio_df = self.compute_portfolio_returns(
            positions, aligned['spy_return']
        )
        
        # Step 5: Compute performance metrics
        metrics = self.compute_performance_metrics(portfolio_df['return'])
        
        # Add turnover
        turnover = self.compute_turnover(positions)
        metrics['turnover'] = turnover
        
        # Store results
        results = {
            'strategy_name': strategy_name,
            'normalization': normalization,
            'position_type': position_type,
            'parameters': {
                'window': window,
                'k': k,
                'w_max': w_max,
                'threshold': threshold,
                'vol_targeting': vol_targeting,
                'target_vol': target_vol
            },
            'positions': positions,
            'z_signal': z_signal,
            'portfolio_returns': portfolio_df,
            'metrics': metrics,
            'dates': aligned.index
        }
        
        self.strategies[strategy_name] = results
        self.results[strategy_name] = metrics
        
        return results
    
    def load_french_factors(self, filepath=None):
        """Load Fama-French factors from CSV file
        
        Handles the standard Fama-French CSV format with header comments and YYYYMMDD date format.
        
        Args:
            filepath: Path to French factors CSV file. If None, returns None.
            
        Returns:
            DataFrame: French factors with date index. Columns normalized to uppercase
                      (e.g., 'Mkt-RF' -> 'MKT-RF')
        """
        if filepath is None:
            print("Warning: No French factors file provided. Factor-neutral analysis skipped.")
            return None
        
        try:
            # Read CSV file, skipping header comments (first 4 lines)
            # The actual header is on line 5 (index 4)
            ff_factors = pd.read_csv(filepath, skiprows=4)
            
            # The first column is unnamed and contains dates in YYYYMMDD format
            # Rename the first column to 'date'
            first_col = ff_factors.columns[0]
            ff_factors = ff_factors.rename(columns={first_col: 'date'})
            
            # Remove any rows that are empty or contain copyright notices
            ff_factors = ff_factors[ff_factors['date'].notna()]
            ff_factors = ff_factors[~ff_factors['date'].astype(str).str.contains('Copyright', na=False)]
            
            # Convert date from YYYYMMDD format to datetime
            # Handle both string and numeric date formats
            date_str = ff_factors['date'].astype(str)
            ff_factors['date'] = pd.to_datetime(date_str, format='%Y%m%d', errors='coerce')
            
            # Remove rows where date conversion failed
            ff_factors = ff_factors[ff_factors['date'].notna()]
            
            # Set date as index
            ff_factors = ff_factors.set_index('date')
            
            # Normalize column names (Mkt-RF -> MKT-RF, etc.)
            # Handle the standard Fama-French column names
            column_mapping = {}
            for col in ff_factors.columns:
                col_upper = col.upper()
                if 'MKT' in col_upper or 'MARKET' in col_upper:
                    column_mapping[col] = 'MKT-RF'
                elif col_upper == 'SMB':
                    column_mapping[col] = 'SMB'
                elif col_upper == 'HML':
                    column_mapping[col] = 'HML'
                elif col_upper == 'RMW':
                    column_mapping[col] = 'RMW'
                elif col_upper == 'CMA':
                    column_mapping[col] = 'CMA'
                elif col_upper == 'RF':
                    column_mapping[col] = 'RF'
                else:
                    column_mapping[col] = col_upper
            
            ff_factors = ff_factors.rename(columns=column_mapping)
            
            # Convert all factor columns to numeric (in case there are any string values)
            for col in ff_factors.columns:
                ff_factors[col] = pd.to_numeric(ff_factors[col], errors='coerce')
            
            # Remove any rows with all NaN values
            ff_factors = ff_factors.dropna(how='all')
            
            # Sort by date
            ff_factors = ff_factors.sort_index()
            
            print(f"Loaded French factors: {len(ff_factors)} observations")
            print(f"Date range: {ff_factors.index.min()} to {ff_factors.index.max()}")
            print(f"Factor columns: {list(ff_factors.columns)}")
            
            return ff_factors
            
        except Exception as e:
            print(f"Error loading French factors: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compute_factor_neutral_returns(self, portfolio_returns, french_factors, 
                                       factors=['MKT-RF', 'SMB', 'HML']):
        """Compute factor-neutral portfolio returns
        
        Args:
            portfolio_returns: Series of portfolio returns
            french_factors: DataFrame of French factors
            factors: List of factor names to neutralize (default: ['MKT-RF', 'SMB', 'HML'])
            
        Returns:
            dict: Factor exposure and neutralized returns
        """
        # Align portfolio returns with factors
        aligned = pd.DataFrame({
            'portfolio_return': portfolio_returns
        })
        
        # Merge with factors
        for factor in factors:
            if factor in french_factors.columns:
                aligned[factor] = french_factors[factor]
        
        aligned = aligned.dropna()
        
        if len(aligned) < 30:  # Need minimum observations for regression
            print("Warning: Insufficient data for factor regression")
            return {
                'factor_exposures': {},
                'alpha': np.nan,
                'alpha_tstat': np.nan,
                'r_squared': np.nan,
                'neutralized_returns': portfolio_returns
            }
        
        # Prepare regression
        y = aligned['portfolio_return'].values
        X = aligned[factors].values
        
        # Run regression with intercept (default behavior)
        # Model: portfolio_return = alpha + β₁*MKT-RF + β₂*SMB + β₃*HML + ε
        reg = LinearRegression(fit_intercept=True)
        reg.fit(X, y)
        
        # Compute statistics
        alpha = reg.intercept_
        factor_exposures = dict(zip(factors, reg.coef_))
        
        # Compute factor-neutral returns: remove only factor exposures, keep alpha
        # Factor-neutral return = portfolio_return - sum(βᵢ * factorᵢ)
        # This gives us alpha + ε, which represents returns orthogonal to factors
        factor_component = np.sum([reg.coef_[i] * X[:, i] for i in range(len(factors))], axis=0)
        neutralized_returns_values = y - factor_component
        
        # Alternative: pure residuals (alpha + ε) - this removes both factors and intercept
        # residuals = y - reg.predict(X)  # This gives pure ε (no alpha, no factors)
        
        # For factor-neutral analysis, we typically want returns with factors removed but alpha included
        # So we use: neutralized_returns = y - factor_component = alpha + ε
        
        # T-statistic for alpha (using standard OLS formula)
        n = len(y)
        k = len(factors)
        residuals = y - reg.predict(X)  # Pure residuals for error calculation
        mse = np.sum(residuals ** 2) / (n - k - 1)  # Unbiased MSE estimator
        
        # Standard error of intercept
        X_with_intercept = np.column_stack([np.ones(n), X])
        XtX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        se_alpha = np.sqrt(mse * XtX_inv[0, 0])
        alpha_tstat = alpha / se_alpha if se_alpha > 0 else np.nan
        
        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        
        # Create neutralized returns series (factor-neutral: alpha + ε)
        neutralized_returns = pd.Series(neutralized_returns_values, index=aligned.index)
        
        return {
            'factor_exposures': factor_exposures,
            'alpha': alpha,
            'alpha_tstat': alpha_tstat,
            'r_squared': r_squared,
            'neutralized_returns': neutralized_returns,
            'regression_model': reg
        }
    
    def analyze_factor_neutral_performance(self, strategy_name, french_factors,
                                          factors=['MKT-RF', 'SMB', 'HML']):
        """Analyze factor-neutral performance for a strategy
        
        Args:
            strategy_name: Name of strategy to analyze
            french_factors: DataFrame of French factors
            factors: List of factor names to neutralize
            
        Returns:
            dict: Factor-neutral analysis results
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found")
        
        strategy = self.strategies[strategy_name]
        portfolio_returns = strategy['portfolio_returns']['return']
        
        # Compute factor-neutral returns
        factor_analysis = self.compute_factor_neutral_returns(
            portfolio_returns, french_factors, factors=factors
        )
        
        # Compute performance metrics on neutralized returns
        neutralized_metrics = self.compute_performance_metrics(factor_analysis['neutralized_returns'])
        
        # Store results
        strategy['factor_neutral'] = {
            'factor_analysis': factor_analysis,
            'neutralized_metrics': neutralized_metrics
        }
        
        return strategy['factor_neutral']
    
    def run_factor_neutral_analysis(self, french_factors_path=None, 
                                   factors=['MKT-RF', 'SMB', 'HML'],
                                   strategy_names=None):
        """Run complete factor-neutral analysis for all strategies
        
        This function loads French factors (if path provided), analyzes factor-neutral
        performance for all strategies, and prints comprehensive results.
        
        Args:
            french_factors_path: Path to French factors CSV file. If None, prints instructions.
            factors: List of factor names to neutralize (default: ['MKT-RF', 'SMB', 'HML'])
            strategy_names: List of strategy names to analyze. If None, analyzes all strategies.
            
        Returns:
            dict: Factor-neutral analysis results for each strategy, or None if factors not available
            
        Example:
            >>> backtester = StrategyBacktester()
            >>> # ... backtest strategies ...
            >>> factor_results = backtester.run_factor_neutral_analysis(
            ...     french_factors_path='data/french_factors.csv'
            ... )
        """
        # Load French factors
        french_factors = self.load_french_factors(french_factors_path)
        
        if french_factors is None:
            print("\n" + "="*60)
            print("FACTOR-NEUTRAL ANALYSIS SKIPPED")
            print("="*60)
            print("To perform factor-neutral analysis:")
            print("1. Download Fama-French factors from:")
            print("   https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html")
            print("2. Save as CSV with columns: date, MKT-RF, SMB, HML, RF")
            print("3. Provide the file path to run_factor_neutral_analysis()")
            return None
        
        # Get strategies to analyze
        if strategy_names is None:
            strategy_names = list(self.strategies.keys())
        
        print("\n" + "="*60)
        print("FACTOR-NEUTRAL ANALYSIS")
        print("="*60)
        
        # Analyze factor-neutral performance for each strategy
        factor_results = {}
        
        for strategy_name in strategy_names:
            if strategy_name not in self.strategies:
                print(f"\nWarning: Strategy '{strategy_name}' not found. Skipping.")
                continue
            
            print(f"\nAnalyzing factor-neutral performance for {strategy_name}...")
            
            try:
                factor_analysis = self.analyze_factor_neutral_performance(
                    strategy_name, 
                    french_factors,
                    factors=factors
                )
                
                factor_results[strategy_name] = factor_analysis
                
                # Print factor exposures
                print(f"\n  Factor Exposures:")
                for factor, exposure in factor_analysis['factor_analysis']['factor_exposures'].items():
                    print(f"    {factor}: {exposure:.4f}")
                
                # Print alpha
                alpha = factor_analysis['factor_analysis']['alpha']
                alpha_tstat = factor_analysis['factor_analysis']['alpha_tstat']
                print(f"\n  Alpha (annualized): {alpha * 252:.2%}")
                print(f"  Alpha t-statistic: {alpha_tstat:.3f}")
                print(f"  R-squared: {factor_analysis['factor_analysis']['r_squared']:.4f}")
                
                # Print neutralized metrics
                neutral_metrics = factor_analysis['neutralized_metrics']
                print(f"\n  Neutralized Performance:")
                print(f"    Sharpe Ratio: {neutral_metrics['sharpe_ratio']:.3f}")
                print(f"    Annualized Return: {neutral_metrics['annualized_return']:.2%}")
                print(f"    Max Drawdown: {neutral_metrics['max_drawdown']:.2%}")
                
            except Exception as e:
                print(f"  Error in factor analysis for {strategy_name}: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*60)
        print("FACTOR-NEUTRAL ANALYSIS COMPLETE")
        print("="*60)
        
        return factor_results
    
    def compare_strategies(self, strategy_names=None):
        """Compare performance across multiple strategies
        
        Args:
            strategy_names: List of strategy names to compare. If None, compares all.
            
        Returns:
            DataFrame: Comparison table
        """
        if strategy_names is None:
            strategy_names = list(self.results.keys())
        
        comparison_data = []
        for name in strategy_names:
            if name in self.results:
                metrics = self.results[name].copy()
                metrics['strategy'] = name
                comparison_data.append(metrics)
        
        if len(comparison_data) == 0:
            return pd.DataFrame()
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Reorder columns
        cols = ['strategy'] + [c for c in comparison_df.columns if c != 'strategy']
        comparison_df = comparison_df[cols]
        
        return comparison_df
    
    def plot_strategy_performance(self, strategy_name, figsize=(16, 10)):
        """Plot comprehensive strategy performance visualization
        
        Args:
            strategy_name: Name of strategy to plot
            figsize: Figure size tuple
            
        Returns:
            fig, axes: Matplotlib figure and axes
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found")
        
        strategy = self.strategies[strategy_name]
        portfolio_df = strategy['portfolio_returns']
        dates = strategy['dates']
        
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle(f'Strategy Performance: {strategy_name}', fontsize=16, fontweight='bold')
        
        # Plot 1: Cumulative returns
        axes[0, 0].plot(dates, portfolio_df['cumulative_return'], label='Strategy', linewidth=2)
        axes[0, 0].plot(dates, (1 + portfolio_df['spy_return']).cumprod(), 
                        label='SPY Buy & Hold', linewidth=1, alpha=0.7, linestyle='--')
        axes[0, 0].set_title('Cumulative Returns', fontweight='bold')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Drawdown
        cumulative = portfolio_df['cumulative_return']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        axes[0, 1].fill_between(dates, drawdown, 0, alpha=0.3, color='red')
        axes[0, 1].plot(dates, drawdown, color='red', linewidth=1)
        axes[0, 1].set_title('Drawdown', fontweight='bold')
        axes[0, 1].set_ylabel('Drawdown')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Positions over time
        axes[1, 0].plot(dates, strategy['positions'], linewidth=1, alpha=0.7)
        axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 0].fill_between(dates, 0, strategy['positions'], 
                                 where=(strategy['positions'] > 0), alpha=0.3, color='green')
        axes[1, 0].fill_between(dates, 0, strategy['positions'], 
                                 where=(strategy['positions'] < 0), alpha=0.3, color='red')
        axes[1, 0].set_title('Position Weights', fontweight='bold')
        axes[1, 0].set_ylabel('Position')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Z-score signal
        axes[1, 1].plot(dates, strategy['z_signal'], linewidth=1, alpha=0.7, color='blue')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 1].set_title('Normalized Signal (Z-score)', fontweight='bold')
        axes[1, 1].set_ylabel('Z-score')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Plot 5: Return distribution
        axes[2, 0].hist(portfolio_df['return'], bins=50, alpha=0.7, edgecolor='black')
        axes[2, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[2, 0].set_title('Return Distribution', fontweight='bold')
        axes[2, 0].set_xlabel('Daily Return')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].grid(True, alpha=0.3)
        
        # Plot 6: Performance metrics table
        axes[2, 1].axis('off')
        metrics = strategy['metrics']
        metrics_text = f"""
        Performance Metrics:
        
        Annualized Return: {metrics['annualized_return']:.2%}
        Annualized Volatility: {metrics['annualized_volatility']:.2%}
        Sharpe Ratio: {metrics['sharpe_ratio']:.3f}
        Max Drawdown: {metrics['max_drawdown']:.2%}
        Calmar Ratio: {metrics['calmar_ratio']:.3f}
        
        Hit Rate: {metrics['hit_rate']:.2%}
        Win Rate: {metrics['win_rate']:.2%}
        Avg Win: {metrics['avg_win']:.4f}
        Avg Loss: {metrics['avg_loss']:.4f}
        Profit Factor: {metrics['profit_factor']:.3f}
        
        Turnover: {metrics['turnover']:.4f}
        """
        axes[2, 1].text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                       verticalalignment='center', transform=axes[2, 1].transAxes)
        
        plt.tight_layout()
        plt.show()
        
        return fig, axes
    
    def plot_strategy_comparison(self, strategy_names=None, figsize=(14, 8)):
        """Plot comparison of multiple strategies
        
        Args:
            strategy_names: List of strategy names to compare. If None, compares all.
            figsize: Figure size tuple
            
        Returns:
            fig, axes: Matplotlib figure and axes
        """
        if strategy_names is None:
            strategy_names = list(self.strategies.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Strategy Comparison', fontsize=16, fontweight='bold')
        
        # Extract metrics for comparison
        comparison_data = []
        for name in strategy_names:
            if name in self.strategies:
                strategy = self.strategies[name]
                metrics = strategy['metrics']
                comparison_data.append({
                    'name': name,
                    'sharpe': metrics['sharpe_ratio'],
                    'return': metrics['annualized_return'],
                    'vol': metrics['annualized_volatility'],
                    'drawdown': metrics['max_drawdown']
                })
        
        if len(comparison_data) == 0:
            print("No strategies to compare")
            return fig, axes
        
        comp_df = pd.DataFrame(comparison_data)
        
        # Plot 1: Sharpe Ratio
        axes[0, 0].barh(range(len(comp_df)), comp_df['sharpe'], color='skyblue')
        axes[0, 0].set_yticks(range(len(comp_df)))
        axes[0, 0].set_yticklabels(comp_df['name'])
        axes[0, 0].set_xlabel('Sharpe Ratio')
        axes[0, 0].set_title('Sharpe Ratio Comparison', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        axes[0, 0].invert_yaxis()
        
        # Plot 2: Return vs Volatility
        axes[0, 1].scatter(comp_df['vol'], comp_df['return'], s=200, alpha=0.7)
        for i, row in comp_df.iterrows():
            axes[0, 1].annotate(row['name'], (row['vol'], row['return']),
                               xytext=(5, 5), textcoords='offset points', fontsize=9)
        axes[0, 1].set_xlabel('Annualized Volatility')
        axes[0, 1].set_ylabel('Annualized Return')
        axes[0, 1].set_title('Return vs Volatility', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Cumulative returns comparison
        for name in strategy_names:
            if name in self.strategies:
                strategy = self.strategies[name]
                dates = strategy['dates']
                cumulative = strategy['portfolio_returns']['cumulative_return']
                axes[1, 0].plot(dates, cumulative, label=name, linewidth=2)
        axes[1, 0].set_title('Cumulative Returns Comparison', fontweight='bold')
        axes[1, 0].set_ylabel('Cumulative Return')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Max Drawdown
        axes[1, 1].barh(range(len(comp_df)), comp_df['drawdown'], color='coral')
        axes[1, 1].set_yticks(range(len(comp_df)))
        axes[1, 1].set_yticklabels(comp_df['name'])
        axes[1, 1].set_xlabel('Max Drawdown')
        axes[1, 1].set_title('Max Drawdown Comparison', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        axes[1, 1].invert_yaxis()
        
        plt.tight_layout()
        plt.show()
        
        return fig, axes


def main():
    """Example usage of StrategyBacktester"""
    print("Strategy Backtesting Module")
    print("="*60)
    print("This module provides tools for:")
    print("1. Normalizing prediction signals (z-score or rank)")
    print("2. Constructing long-short portfolios")
    print("3. Backtesting strategy performance")
    print("4. Factor-neutral analysis")
    print("="*60)


if __name__ == "__main__":
    main()

