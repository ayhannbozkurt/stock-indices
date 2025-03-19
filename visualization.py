import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import seaborn as sns

def plot_normalized_prices(df_normalized):
    """Visualizes normalized price movements of all banks"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use more colors to distinguish all banks
    colors = list(mcolors.TABLEAU_COLORS) + list(mcolors.CSS4_COLORS)
    color_idx = 0
    
    for banka in df_normalized.columns:
        if banka != 'XBANK':
            ax.plot(df_normalized.index, df_normalized[banka], label=banka, alpha=0.7, color=colors[color_idx % len(colors)])
            color_idx += 1
    
    ax.plot(df_normalized.index, df_normalized['XBANK'], label='Market Index', linewidth=3, color='black')
    ax.set_title('Normalized Price Comparison (Starting Value=100)')
    ax.legend(loc='upper left', fontsize='small')
    ax.grid(True)
    return fig

def plot_correlations(sonuclar):
    """Visualizes correlation comparison with the market index"""
    bankalar = [banka for banka in sonuclar.keys() if banka != 'XBANK_perf']
    korelasyonlar = [sonuclar[banka]['correlation'] for banka in bankalar]
    # Sort by correlation
    sort_indices = np.argsort(korelasyonlar)[::-1]  # Descending order
    korelasyonlar_sorted = [korelasyonlar[i] for i in sort_indices]
    bankalar_sorted = [bankalar[i] for i in sort_indices]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(bankalar_sorted, korelasyonlar_sorted)
    
    # Apply color gradient
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.viridis(i/len(bars)))
    
    ax.set_title('Correlation with Market Index (Descending)')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, axis='y')
    return fig, bankalar_sorted

def plot_betas(sonuclar):
    """Visualizes beta comparison with the market index"""
    bankalar = [banka for banka in sonuclar.keys() if banka != 'XBANK_perf']
    betalar = [sonuclar[banka]['beta'] for banka in bankalar]
    # Sort by beta
    sort_indices = np.argsort(betalar)[::-1]  # Descending order
    betalar_sorted = [betalar[i] for i in sort_indices]
    bankalar_beta_sorted = [bankalar[i] for i in sort_indices]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(bankalar_beta_sorted, betalar_sorted)
    
    # Apply color gradient
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.plasma(i/len(bars)))
    
    ax.set_title('Beta with Market Index (Descending)')
    ax.axhline(y=1, color='r', linestyle='--', label='Beta=1')
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, axis='y')
    ax.legend()
    return fig, bankalar_beta_sorted

def plot_performance(sonuclar):
    """Visualizes 1-year performance of banks"""
    bankalar = [banka for banka in sonuclar.keys() if banka != 'XBANK_perf']
    performanslar = [sonuclar[banka]['perf'] for banka in bankalar]
    # Sort by performance
    sort_indices = np.argsort(performanslar)[::-1]  # Descending order
    performanslar_sorted = [performanslar[i] for i in sort_indices]
    bankalar_perf_sorted = [bankalar[i] for i in sort_indices]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(bankalar_perf_sorted, performanslar_sorted)
    
    # Apply color gradient based on performance values
    norm = plt.Normalize(min(performanslar_sorted), max(performanslar_sorted))
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.coolwarm(norm(performanslar_sorted[i])))
    
    xbank_perf = sonuclar['XBANK_perf']
    ax.axhline(y=xbank_perf, color='r', linestyle='--', label=f'Market Index: {xbank_perf:.2f}%')
    ax.set_title('1-Year Performance (%)')
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, axis='y')
    ax.legend()
    return fig, bankalar_perf_sorted

def plot_rolling_correlation(df_returns, bankalar, window_size=30):
    """Visualizes rolling correlation of selected banks with the market index"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create color palette
    colors = plt.cm.rainbow(np.linspace(0, 1, len(bankalar)))
    
    for i, banka in enumerate(bankalar):
        rolling_corr = df_returns[banka].rolling(window=window_size).corr(df_returns['XBANK'])
        ax.plot(rolling_corr.index, rolling_corr, label=banka, alpha=0.7, linewidth=1.5, color=colors[i])
    
    ax.set_title(f'{window_size}-Day Rolling Correlation with Market Index')
    # Place legend outside for better readability
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax.grid(True)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    return fig

def plot_relative_performance(df_normalized, bankalar):
    """Visualizes performance of banks relative to the market index"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create color palette
    colors = plt.cm.viridis(np.linspace(0, 1, len(bankalar)))
    
    for i, banka in enumerate(bankalar):
        relative_perf = df_normalized[banka] / df_normalized['XBANK'] * 100
        ax.plot(df_normalized.index, relative_perf, label=banka, alpha=0.7, linewidth=1.5, color=colors[i])
    
    ax.set_title('Bank/Market Index Performance Ratio')
    ax.axhline(y=100, color='r', linestyle='-', alpha=0.3, label='Market Index')
    # Place legend outside for better readability
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax.grid(True)
    plt.tight_layout()
    return fig

def plot_cumulative_diff(df_normalized, bankalar):
    """Visualizes cumulative performance difference from the market index"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create color palette
    colors = plt.cm.plasma(np.linspace(0, 1, len(bankalar)))
    
    for i, banka in enumerate(bankalar):
        cumulative_diff = (df_normalized[banka] - df_normalized['XBANK'])
        ax.plot(df_normalized.index, cumulative_diff, label=banka, alpha=0.7, linewidth=1.5, color=colors[i])
    
    ax.set_title('Bank - Market Index Normalized Price Difference')
    ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    # Place legend outside for better readability
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax.grid(True)
    plt.tight_layout()
    return fig

def plot_beta_performance(sonuclar):
    """Visualizes the relationship between market index, beta, and performance"""
    bankalar = [banka for banka in sonuclar.keys() if banka != 'XBANK_perf']
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color by correlation values
    korelasyonlar = [sonuclar[banka]['correlation'] for banka in bankalar]
    scatter = ax.scatter(
        [sonuclar[banka]['beta'] for banka in bankalar], 
        [sonuclar[banka]['perf'] for banka in bankalar], 
        alpha=0.7, s=100, c=korelasyonlar, cmap='viridis'
    )
    
    # Add color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Correlation')

    # Add bank names to data points
    for banka in bankalar:
        ax.annotate(banka, 
                   (sonuclar[banka]['beta'], sonuclar[banka]['perf']),
                   xytext=(5, 5), textcoords='offset points')

    ax.set_title('Beta - Performance Relationship')
    ax.set_xlabel('Beta Coefficient')
    ax.set_ylabel('1-Year Performance (%)')
    ax.axhline(y=sonuclar['XBANK_perf'], color='r', linestyle='--', 
               label=f'Market Index: {sonuclar["XBANK_perf"]:.2f}%')
    ax.axvline(x=1, color='g', linestyle='--', label='Beta = 1')
    ax.grid(True)
    ax.legend()
    return fig

def plot_structural_breaks(df_normalized, break_results):
    """Visualizes structural breaks in time series data"""
    target_col = break_results['target']
    break_dates = break_results['break_dates']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot target series
    ax.plot(df_normalized.index, df_normalized[target_col], label=target_col if target_col != 'XBANK' else 'Market Index', linewidth=2, color='blue')
    
    # Mark breakpoints
    for date in break_dates:
        if date in df_normalized.index:
            ax.axvline(x=date, color='red', linestyle='--', alpha=0.7)
            # Add date label
            ax.text(date, df_normalized[target_col].max() * 1.05, 
                   date.strftime('%Y-%m-%d'), 
                   rotation=45, ha='right')
    
    # Color segments
    segments = break_results['segments']
    colors = plt.cm.tab10(np.linspace(0, 1, len(segments)))
    
    for i, segment in enumerate(segments):
        start_date = segment['start_date']
        end_date = segment['end_date']
        trend = segment['trend_pct']
        
        # Define segment range
        segment_idx = (df_normalized.index >= start_date) & (df_normalized.index <= end_date)
        
        # Add transparent color over segment
        ax.fill_between(df_normalized.index[segment_idx], 
                       0, df_normalized[target_col].max() * 1.1,
                       alpha=0.1, color=colors[i % len(colors)])
        
        # Show trend direction
        mid_point = df_normalized.index[segment_idx].mean()
        trend_text = f"{trend:.1f}%"
        y_pos = df_normalized[target_col].min() * 0.9
        
        # Set color based on trend direction
        text_color = 'green' if trend > 0 else 'red'
        ax.text(mid_point, y_pos, trend_text, color=text_color, 
               ha='center', va='bottom', fontweight='bold')
    
    title_text = 'Market Index Structural Break Analysis' if target_col == 'XBANK' else f'{target_col} Structural Break Analysis'
    ax.set_title(title_text)
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Value (100)')
    ax.grid(True)
    plt.tight_layout()
    
    return fig

def plot_cluster_analysis(df_normalized, clusters, metrics):
    """Visualizes cluster analysis results"""
    # 1. Show cluster members
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    # Prepare data for bar chart
    cluster_counts = {f"Cluster {i+1}": len(banks) for i, banks in clusters.items()}
    x = list(cluster_counts.keys())
    y = list(cluster_counts.values())
    
    # Draw bars
    bars = ax1.bar(x, y)
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f"{height}", ha='center', va='bottom')
    
    ax1.set_title("Cluster Analysis - Cluster Sizes")
    ax1.set_xlabel("Cluster")
    ax1.set_ylabel("Number of Banks")
    
    # 2. 2D visualization with PCA - Use correlation matrix
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    # Cluster colors
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))
    
    # Plot for each bank
    for cluster_id, banks in clusters.items():
        # For each cluster, take the average of normalized prices of banks in the cluster
        cluster_data = df_normalized[banks].mean(axis=1)
        ax2.plot(df_normalized.index, cluster_data, label=f"Cluster {cluster_id+1}", 
                linewidth=2, color=cluster_colors[cluster_id % len(cluster_colors)])
    
    # Add market index
    ax2.plot(df_normalized.index, df_normalized['XBANK'], label='Market Index', 
            linewidth=3, color='black', linestyle='--')
    
    ax2.set_title("Bank Clusters - Average Price Movements")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Normalized Value (100)")
    ax2.legend()
    ax2.grid(True)
    
    # 3. Heatmap showing cluster properties
    cluster_centers = metrics['cluster_centers']
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(cluster_centers, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax3)
    ax3.set_title("Cluster Centers - Properties")
    
    # 4. Correlation matrix and clustering
    fig4, ax4 = plt.subplots(figsize=(12, 10))
    
    # Get daily returns for all banks except market index
    corr_matrix = df_normalized.drop(columns=['XBANK']).pct_change().dropna().corr()
    
    # Order by clusters
    ordered_banks = []
    for cluster_id in sorted(clusters.keys()):
        ordered_banks.extend(clusters[cluster_id])
    
    # Ordered correlation matrix
    corr_ordered = corr_matrix.loc[ordered_banks, ordered_banks]
    
    # Heatmap
    mask = np.zeros_like(corr_ordered)
    mask[np.triu_indices_from(mask)] = True
    
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr_ordered, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
               square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax4)
    
    # Show cluster boundaries
    current_idx = 0
    for cluster_id in sorted(clusters.keys()):
        cluster_size = len(clusters[cluster_id])
        # Draw rectangle
        rect = plt.Rectangle((current_idx, current_idx), cluster_size, cluster_size, 
                          fill=False, linewidth=2, edgecolor='red')
        ax4.add_patch(rect)
        current_idx += cluster_size
    
    ax4.set_title("Correlation Matrix of Bank Returns (Ordered by Clusters)")
    
    return [fig1, fig2, fig3, fig4]

def plot_cluster_members(clusters, selected_cluster=None):
    """Visualizes cluster members in a table format"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    if selected_cluster is not None and selected_cluster in clusters:
        # Show only selected cluster
        table_data = [clusters[selected_cluster]]
        row_labels = [f"Cluster {selected_cluster+1}"]
    else:
        # Show all clusters
        table_data = [clusters[i] for i in sorted(clusters.keys())]
        row_labels = [f"Cluster {i+1}" for i in sorted(clusters.keys())]
    
    # Find longest list and pad others with empty strings
    max_len = max(len(cluster) for cluster in table_data)
    padded_data = [cluster + [''] * (max_len - len(cluster)) for cluster in table_data]
    
    # Create table
    table = ax.table(cellText=padded_data, rowLabels=row_labels,
                    loc='center', cellLoc='center')
    
    # Table formatting
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Title
    if selected_cluster is not None:
        ax.set_title(f"Banks in Cluster {selected_cluster+1}", fontsize=14)
    else:
        ax.set_title("Bank Clusters", fontsize=14)
    
    return fig

def plot_performance_heatmap(df_returns, bankalar):
    """
    Creates a heatmap showing monthly performance of banks.
    Color scale: Red (negative) - White (neutral) - Green (positive)
    """
    # Calculate monthly returns
    monthly_returns = df_returns.copy()
    monthly_returns.index = pd.to_datetime(monthly_returns.index)
    monthly_returns = monthly_returns.resample('M').apply(
        lambda x: ((1 + x).prod() - 1) * 100  # Monthly return in percentage
    )
    
    # Select only banks
    monthly_bank_returns = monthly_returns[bankalar].copy()
    # Format months
    monthly_bank_returns.index = monthly_bank_returns.index.strftime('%Y-%m')
    
    # Transpose (banks as rows, months as columns)
    data = monthly_bank_returns.transpose()
    
    # Visualization
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Heatmap
    cmap = sns.diverging_palette(10, 130, as_cmap=True)  # Red-Green
    heatmap = sns.heatmap(data, cmap=cmap, center=0, annot=True, fmt=".1f", 
                          linewidths=.5, ax=ax, cbar_kws={'label': 'Monthly Return (%)'})
    
    # Title and axis labels
    ax.set_title('Monthly Performance of Banks (%)', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Bank', fontsize=12)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    return fig 