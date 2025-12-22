"""
Data exploration and statistics functions for news dataset
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class DataExplorer:
    """Class for computing statistics and creating visualizations for the news dataset."""
    
    def compute_news_statistics(self, df_news):
        """Compute comprehensive statistics about the news dataset"""
        print("="*60)
        print("NEWS DATASET STATISTICS")
        print("="*60)
        
        # Calculate headline token counts
        df_news['headline_tokens'] = df_news['headline'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
        df_news['headline_length'] = df_news['headline'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
        
        # Basic statistics
        print(f"\nDataset Overview:")
        print(f"- Total articles: {len(df_news):,}")
        print(f"- Date range: {df_news['date'].min()} to {df_news['date'].max()}")
        print(f"- Number of unique categories: {df_news['category'].nunique()}")
        print(f"- Number of unique dates: {df_news['date'].nunique()}")
        
        print(f"\nHeadline Statistics:")
        print(f"- Average tokens per headline: {df_news['headline_tokens'].mean():.1f}")
        print(f"- Median tokens per headline: {df_news['headline_tokens'].median():.1f}")
        print(f"- Average characters per headline: {df_news['headline_length'].mean():.1f}")
        
        # News per day
        daily_counts = df_news.groupby('date').size().reset_index(name='count')
        daily_counts = daily_counts.sort_values('date')
        
        print(f"\nDaily News Statistics:")
        print(f"- Average news per day: {daily_counts['count'].mean():.1f}")
        print(f"- Median news per day: {daily_counts['count'].median():.1f}")
        print(f"- Max news in a single day: {daily_counts['count'].max()}")
        print(f"- Min news in a single day: {daily_counts['count'].min()}")
        
        # News per month
        df_news['year_month'] = df_news['date'].dt.to_period('M')
        monthly_counts = df_news.groupby('year_month').size().reset_index(name='count')
        monthly_counts['year_month'] = monthly_counts['year_month'].astype(str)
        
        print(f"\nMonthly News Statistics:")
        print(f"- Average news per month: {monthly_counts['count'].mean():.1f}")
        print(f"- Total months covered: {len(monthly_counts)}")
        
        # Category statistics
        category_counts = df_news['category'].value_counts()
        print(f"\nTop 15 Categories:")
        print(category_counts.head(15))
        
        return df_news, daily_counts, monthly_counts, category_counts

    def plot_news_per_day(self, ax, daily_counts):
        """Plot news articles per day over time"""
        ax.plot(daily_counts['date'], daily_counts['count'], linewidth=1, alpha=0.7)
        ax.set_title('Total Number of News Articles Per Day Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Number of Articles', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Add rolling average
        daily_counts['rolling_30d'] = daily_counts['count'].rolling(window=30, min_periods=1).mean()
        ax.plot(daily_counts['date'], daily_counts['rolling_30d'], 
                 color='red', linewidth=2, label='30-day Moving Average')
        ax.legend()

    def plot_category_monthly_stacked(self, ax, df_news, category_counts):
        """Plot news articles per category per month as stacked bar chart"""
        # Prepare data: group by month and category
        monthly_category_counts = df_news.groupby(['year_month', 'category']).size().reset_index(name='count')
        monthly_category_counts['year_month'] = monthly_category_counts['year_month'].astype(str)
        
        # Get top categories (top 10 for readability)
        top_cats = category_counts.head(10).index.tolist()
        
        # Create pivot table: months as index, categories as columns
        monthly_category_pivot = monthly_category_counts.pivot_table(
            index='year_month', 
            columns='category', 
            values='count', 
            fill_value=0
        )
        
        # Sort by year_month
        monthly_category_pivot = monthly_category_pivot.sort_index()
        
        # Select only top categories for the plot
        top_cats_in_data = [cat for cat in top_cats if cat in monthly_category_pivot.columns]
        monthly_category_plot = monthly_category_pivot[top_cats_in_data]
        
        # Create stacked bar chart
        x_positions = range(len(monthly_category_plot))
        bottom = np.zeros(len(monthly_category_plot))
        
        # Use a colormap for better distinction
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_cats_in_data)))
        
        for i, cat in enumerate(top_cats_in_data):
            ax.bar(x_positions, monthly_category_plot[cat], bottom=bottom, 
                    label=cat, color=colors[i], alpha=0.8)
            bottom += monthly_category_plot[cat].values
        
        ax.set_title('News Articles Per Category Per Month (Stacked)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Number of Articles', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set x-axis labels (show every 12 months for readability)
        n_ticks = min(12, len(monthly_category_plot))
        tick_positions = np.linspace(0, len(monthly_category_plot)-1, n_ticks).astype(int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([monthly_category_plot.index[i] for i in tick_positions], 
                             rotation=45, ha='right')
        
        # Add legend (place it outside the plot area)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)

    def plot_category_distribution(self, ax, category_counts):
        """Plot top categories by article count"""
        top_categories = category_counts.head(15)
        ax.barh(range(len(top_categories)), top_categories.values, color='coral', alpha=0.7)
        ax.set_yticks(range(len(top_categories)))
        ax.set_yticklabels(top_categories.index)
        ax.set_title('Top 15 News Categories by Article Count', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Articles', fontsize=12)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

    def plot_token_distribution(self, ax, df_news):
        """Plot distribution of headline token counts"""
        # Filter out extreme outliers for better visualization
        token_data = df_news['headline_tokens'][df_news['headline_tokens'] <= df_news['headline_tokens'].quantile(0.99)]
        ax.hist(token_data, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
        ax.axvline(df_news['headline_tokens'].mean(), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {df_news["headline_tokens"].mean():.1f}')
        ax.axvline(df_news['headline_tokens'].median(), color='green', linestyle='--', 
                    linewidth=2, label=f'Median: {df_news["headline_tokens"].median():.1f}')
        ax.set_title('Distribution of Headline Token Counts', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Tokens (words)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    def plot_avg_token_length_monthly(self, ax, df_news):
        """Plot average headline token length per month"""
        # Calculate average tokens per month
        monthly_avg_tokens = df_news.groupby('year_month')['headline_tokens'].mean().reset_index()
        monthly_avg_tokens['year_month'] = monthly_avg_tokens['year_month'].astype(str)
        monthly_avg_tokens = monthly_avg_tokens.sort_values('year_month')
        
        # Create the plot
        x_positions = range(len(monthly_avg_tokens))
        ax.plot(x_positions, monthly_avg_tokens['headline_tokens'], 
                marker='o', linewidth=2, markersize=4, color='steelblue', alpha=0.7)
        
        # Add horizontal line for overall average
        overall_avg = df_news['headline_tokens'].mean()
        ax.axhline(y=overall_avg, color='red', linestyle='--', linewidth=2, 
                   label=f'Overall Average: {overall_avg:.2f} tokens', alpha=0.7)
        
        # Add 12-month moving average
        monthly_avg_tokens['rolling_12m'] = monthly_avg_tokens['headline_tokens'].rolling(window=12, min_periods=1).mean()
        ax.plot(x_positions, monthly_avg_tokens['rolling_12m'], 
                color='green', linewidth=2, linestyle='--', 
                label='12-Month Moving Average', alpha=0.8)
        
        ax.set_title('Average Headline Token Length Per Month', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Average Number of Tokens', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Set x-axis labels (show every 12 months for readability)
        n_ticks = min(12, len(monthly_avg_tokens))
        tick_positions = np.linspace(0, len(monthly_avg_tokens)-1, n_ticks).astype(int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([monthly_avg_tokens.iloc[i]['year_month'] for i in tick_positions], 
                            rotation=45, ha='right')
        
        return overall_avg, monthly_avg_tokens

    def plot_news_statistics(self, df_news, daily_counts, monthly_counts, category_counts):
        """Create comprehensive visualizations of news dataset statistics"""
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # Plot 1: News per day over time
        self.plot_news_per_day(axes[0, 0], daily_counts)
        
        # Plot 2: News per category per month (stacked bar chart)
        self.plot_category_monthly_stacked(axes[0, 1], df_news, category_counts)
        
        # Plot 3: Category distribution (top 15)
        self.plot_category_distribution(axes[1, 0], category_counts)
        
        # Plot 4: Headline token distribution
        self.plot_token_distribution(axes[1, 1], df_news)
        
        # Plot 5: Average headline token length per month
        overall_avg, monthly_avg_tokens = self.plot_avg_token_length_monthly(axes[2, 0], df_news)
        
        # Plot 6: Empty or additional plot (can be used for future visualizations)
        axes[2, 1].axis('off')  # Hide empty subplot
        
        plt.tight_layout()
        plt.show()
        
        print("\nVisualizations created!")
        print(f"\nAverage headline token length statistics:")
        print(f"- Overall average: {overall_avg:.2f} tokens")
        print(f"- Monthly average range: {monthly_avg_tokens['headline_tokens'].min():.2f} - {monthly_avg_tokens['headline_tokens'].max():.2f} tokens")
        print(f"- Monthly average std: {monthly_avg_tokens['headline_tokens'].std():.2f} tokens")
