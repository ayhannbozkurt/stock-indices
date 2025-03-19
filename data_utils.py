import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import ruptures as rpt

def fetch_bank_data(days=365):
    """
    Fetches bank data for the specified number of days.

    Args:
        days (int): Number of days to fetch data for.

    Returns:
        pd.DataFrame: Combined bank data.
    """
    # Define date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)  # Get data for specified number of days
    
    # List of all bank stocks and their names
    banka_hisseleri = {
        "AKBNK.IS": "AKBANK",
        "ALBRK.IS": "ALBARAKA TÜRK",
        "ICBCT.IS": "ICBC TURKEY",
        "QNBTR.IS": "QNB FINANSBANK",
        "SKBNK.IS": "ŞEKERBANK",
        "GARAN.IS": "GARANTİ BANKASI",
        "HALKB.IS": "HALKBANK",
        "ISCTR.IS": "İŞ BANKASI C",
        "KLNMA.IS": "KALKINMA BANKASI",
        "TSKB.IS": "SINAİ KALKINMA",
        "VAKBN.IS": "VAKIFBANK",
        "YKBNK.IS": "YAPI KREDİ"
    }
    
    # Download XBANK index
    print("Downloading XBANK index...")
    xbank = yf.download("XBANK.IS", start=start_date, end=end_date)
    
    # Flatten MultiIndex columns
    if isinstance(xbank.columns, pd.MultiIndex):
        xbank.columns = xbank.columns.get_level_values(0)
    
    # Collect all bank data in a dictionary
    banka_verileri = {}
    for sembol, isim in banka_hisseleri.items():
        try:
            print(f"Downloading {isim} data...")
            veri = yf.download(sembol, start=start_date, end=end_date)
            
            # Flatten MultiIndex columns
            if isinstance(veri.columns, pd.MultiIndex):
                veri.columns = veri.columns.get_level_values(0)
            
            # Check for empty data
            if len(veri) > 0:
                banka_verileri[isim] = veri['Close']
                print(f"{isim} data downloaded. Data size: {len(veri)}")
            else:
                print(f"No data found for {isim}.")
        except Exception as e:
            print(f"Error downloading {isim} data: {str(e)}")
    
    # Combine all data into a single DataFrame
    df = pd.DataFrame(banka_verileri)
    df['XBANK'] = xbank['Close']
    
    # Clean missing data
    df = df.dropna()
    print(f"Combined data size: {df.shape}")
    
    # Filter banks with at least 100 data points
    yeterli_veri_olan_bankalar = [kolon for kolon in df.columns if kolon != 'XBANK' and df[kolon].count() >= 100]
    print(f"Number of banks with sufficient data: {len(yeterli_veri_olan_bankalar)}")
    df = df[yeterli_veri_olan_bankalar + ['XBANK']]
    
    return df

def analyze_bank_data(df):
    """
    Analyzes bank data and returns normalized data, daily returns, and results.

    Args:
        df (pd.DataFrame): Bank data.

    Returns:
        tuple: Normalized data, daily returns, and results.
    """
    # Normalize (starting value = 100)
    df_normalized = df.copy()
    for kolon in df_normalized.columns:
        df_normalized[kolon] = df_normalized[kolon] / df_normalized[kolon].iloc[0] * 100
    
    # Daily returns
    df_returns = df.pct_change().dropna()
    
    sonuclar = {}
    for banka in df.columns:
        if banka != 'XBANK':
            correlation, p_value = pearsonr(df_returns[banka], df_returns['XBANK'])
            
            # Calculate beta
            beta = np.cov(df_returns[banka], df_returns['XBANK'])[0, 1] / np.var(df_returns['XBANK'])
            
            # Total performance (entire period)
            banka_perf = (df_normalized[banka].iloc[-1] / df_normalized[banka].iloc[0] - 1) * 100
            xbank_perf = (df_normalized['XBANK'].iloc[-1] / df_normalized['XBANK'].iloc[0] - 1) * 100
            
            sonuclar[banka] = {
                'correlation': correlation,
                'p_value': p_value,
                'beta': beta,
                'perf': banka_perf
            }
            
    # Total performance for XBANK
    xbank_perf = (df_normalized['XBANK'].iloc[-1] / df_normalized['XBANK'].iloc[0] - 1) * 100
    sonuclar['XBANK_perf'] = xbank_perf
    
    return df_normalized, df_returns, sonuclar

def cluster_banks(df_returns):
    """
    Groups banks with similar movement patterns using cluster analysis.

    Args:
        df_returns (pd.DataFrame): Daily returns.

    Returns:
        tuple: Clusters and metrics.
    """
    # Analyze excluding XBANK
    df_cluster = df_returns.drop(columns=['XBANK'])
    
    # Clean missing values
    df_cluster = df_cluster.dropna(axis=1)
    
    # Prepare data for clustering
    features = pd.DataFrame(index=df_cluster.columns)
    
    # Volatility (standard deviation)
    features['volatility'] = df_cluster.std()
    
    # Correlation with XBANK
    features['xbank_corr'] = [df_returns[col].corr(df_returns['XBANK']) for col in df_cluster.columns]
    
    # Beta
    features['beta'] = [
        np.cov(df_returns[col], df_returns['XBANK'])[0, 1] / np.var(df_returns['XBANK']) 
        for col in df_cluster.columns
    ]
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Determine optimal number of clusters (Elbow method)
    distortions = []
    K_range = range(1, min(8, len(df_cluster.columns)))
    for k in K_range:
        if k < len(df_cluster.columns):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(features_scaled)
            distortions.append(kmeans.inertia_)
    
    # If not enough data
    if len(K_range) <= 1:
        optimal_k = 1
    else:
        # Simple elbow method - check slope change
        deltas = np.diff(distortions)
        if len(deltas) > 0:
            optimal_k = np.argmax(deltas) + 1
            # Minimum 2 clusters
            optimal_k = max(2, optimal_k)
            # Not more than half the number of banks
            optimal_k = min(optimal_k, len(df_cluster.columns) // 2)
        else:
            optimal_k = 1
    
    # Apply KMeans
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters_pred = kmeans.fit_predict(features_scaled)
    
    # Organize results
    clusters = {}
    for i, bank in enumerate(df_cluster.columns):
        cluster_id = int(clusters_pred[i])
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(bank)
    
    # Transform cluster centers back to original features
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    center_df = pd.DataFrame(
        cluster_centers, 
        columns=features.columns,
        index=[f"Cluster {i+1}" for i in range(optimal_k)]
    )
    
    # Correlation of each cluster with XBANK
    cluster_xbank_corr = {}
    for cluster_id, banks in clusters.items():
        # Calculate average return for each cluster
        cluster_returns = df_cluster[banks].mean(axis=1)
        # Correlation with XBANK
        corr = np.corrcoef(cluster_returns, df_returns['XBANK'])[0, 1]
        cluster_xbank_corr[cluster_id] = corr
    
    # Prepare metrics
    metrics = {
        'optimal_k': optimal_k,
        'distortions': distortions,
        'cluster_centers': center_df,
        'cluster_xbank_corr': cluster_xbank_corr
    }
    
    return clusters, metrics

def detect_structural_breaks(series, min_size=30, n_bkps=5):
    """
    Detects structural breaks in a time series.

    Args:
        series (pd.Series): Time series.
        min_size (int): Minimum segment size.
        n_bkps (int): Number of breaks.

    Returns:
        tuple: Break dates and algorithm.
    """
    # Clean missing data
    clean_series = series.dropna()
    
    # Prepare data for Ruptures algorithm
    signal = clean_series.values.reshape(-1, 1)
    
    # Algorithm
    # For PELT algorithm, we need to use penalty parameter instead of n_bkps
    algo = rpt.Pelt(model="l2", min_size=min_size).fit(signal)
    
    # Calculate appropriate pen value for desired number of breaks
    # Pelt.predict() method takes pen parameter, not n_bkps
    # We can find pen value manually
    pen_min = 1
    pen_max = 10000
    pen = 1000  # Initial value
    
    # Find appropriate pen value using binary search
    # This value will give us the desired number of break points
    bkps = algo.predict(pen=pen)
    for _ in range(10):  # Maximum 10 attempts
        if len(bkps) - 1 < n_bkps:  # Too few breaks, decrease pen
            pen_max = pen
            pen = (pen_min + pen)/2
        elif len(bkps) - 1 > n_bkps:  # Too many breaks, increase pen
            pen_min = pen
            pen = (pen + pen_max)/2
        else:
            break  # Found desired number of break points
        bkps = algo.predict(pen=pen)
    
    # n_bkps+1 points, first and last are start and end
    # if we can't find the exact number, return the closest result
    
    # Convert to original index
    bkps_dates = [clean_series.index[i-1] if i < len(clean_series) else clean_series.index[-1] for i in bkps if i > 0]
    
    return bkps_dates, algo

def analyze_structural_breaks(df, target_col="XBANK", n_bkps=5):
    """
    Analyzes structural breaks and returns results.

    Args:
        df (pd.DataFrame): Data.
        target_col (str): Target column.
        n_bkps (int): Number of breaks.

    Returns:
        dict: Results.
    """
    # Use normalized data
    df_norm = df.copy()
    for col in df_norm.columns:
        df_norm[col] = df_norm[col] / df_norm[col].iloc[0] * 100
    
    # Detect structural breaks
    break_dates, algo = detect_structural_breaks(
        df_norm[target_col], 
        min_size=30, 
        n_bkps=n_bkps
    )
    
    # Prepare results
    break_results = {
        'target': target_col,
        'break_dates': break_dates,
        'algorithm': algo,
        'segments': []
    }
    
    # Calculate statistics for each segment
    segments = []
    prev_date = df_norm.index[0]
    
    for date in break_dates:
        if date <= prev_date:
            continue
            
        segment_data = df_norm.loc[prev_date:date]
        
        if len(segment_data) < 5:  # Skip very short segments
            continue
            
        # Segment start and end values
        start_val = segment_data[target_col].iloc[0]
        end_val = segment_data[target_col].iloc[-1]
        
        # Trend for segment (percent change)
        trend_pct = ((end_val / start_val) - 1) * 100
        
        # Volatility (standard deviation)
        volatility = df.loc[prev_date:date, target_col].pct_change().std() * 100
        
        segment_info = {
            'start_date': prev_date,
            'end_date': date,
            'duration_days': (date - prev_date).days,
            'trend_pct': trend_pct,
            'volatility': volatility,
            'avg_daily_change': trend_pct / max(1, (date - prev_date).days),
        }
        
        segments.append(segment_info)
        prev_date = date
    
    # Last segment
    if prev_date < df_norm.index[-1]:
        segment_data = df_norm.loc[prev_date:]
        
        if len(segment_data) >= 5:
            start_val = segment_data[target_col].iloc[0]
            end_val = segment_data[target_col].iloc[-1]
            trend_pct = ((end_val / start_val) - 1) * 100
            volatility = df.loc[prev_date:, target_col].pct_change().std() * 100
            
            segment_info = {
                'start_date': prev_date,
                'end_date': df_norm.index[-1],
                'duration_days': (df_norm.index[-1] - prev_date).days,
                'trend_pct': trend_pct,
                'volatility': volatility,
                'avg_daily_change': trend_pct / max(1, (df_norm.index[-1] - prev_date).days),
            }
            
            segments.append(segment_info)
    
    break_results['segments'] = segments
    
    return break_results