import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import ruptures as rpt

def fetch_bank_data(days=365):
    # Tarih aralığını belirle
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)  # İstenilen gün sayısı kadar veri
    
    # Bütün banka hisselerinin listesi ve isimleri
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
    
    # XBANK endeksi indir
    print("XBANK endeksi indiriliyor...")
    xbank = yf.download("XBANK.IS", start=start_date, end=end_date)
    
    # MultiIndex sütunlarını düzleştir
    if isinstance(xbank.columns, pd.MultiIndex):
        xbank.columns = xbank.columns.get_level_values(0)
    
    # Tüm banka verilerini bir sözlükte topla
    banka_verileri = {}
    for sembol, isim in banka_hisseleri.items():
        try:
            print(f"{isim} verisi indiriliyor...")
            veri = yf.download(sembol, start=start_date, end=end_date)
            
            # MultiIndex sütunlarını düzleştir
            if isinstance(veri.columns, pd.MultiIndex):
                veri.columns = veri.columns.get_level_values(0)
            
            # Boş veri kontrolü
            if len(veri) > 0:
                banka_verileri[isim] = veri['Close']
                print(f"{isim} verisi indirildi. Veri boyutu: {len(veri)}")
            else:
                print(f"{isim} için veri bulunamadı.")
        except Exception as e:
            print(f"{isim} verisi indirilirken hata: {str(e)}")
    
    # Tüm verileri tek bir DataFrame'de birleştir
    df = pd.DataFrame(banka_verileri)
    df['XBANK'] = xbank['Close']
    
    # Eksik verileri temizle
    df = df.dropna()
    print(f"Birleştirilmiş veri boyutu: {df.shape}")
    
    # En az 100 veri noktası olan bankaları filtrele
    yeterli_veri_olan_bankalar = [kolon for kolon in df.columns if kolon != 'XBANK' and df[kolon].count() >= 100]
    print(f"Yeterli verisi olan banka sayısı: {len(yeterli_veri_olan_bankalar)}")
    df = df[yeterli_veri_olan_bankalar + ['XBANK']]
    
    return df

def analyze_bank_data(df):
    # Normalize et (başlangıç değeri 100)
    df_normalized = df.copy()
    for kolon in df_normalized.columns:
        df_normalized[kolon] = df_normalized[kolon] / df_normalized[kolon].iloc[0] * 100
    
    # Günlük değişimler
    df_returns = df.pct_change().dropna()
    
    sonuclar = {}
    for banka in df.columns:
        if banka != 'XBANK':
            correlation, p_value = pearsonr(df_returns[banka], df_returns['XBANK'])
            
            # Beta hesapla
            beta = np.cov(df_returns[banka], df_returns['XBANK'])[0, 1] / np.var(df_returns['XBANK'])
            
            # Toplam performans (tüm dönem)
            banka_perf = (df_normalized[banka].iloc[-1] / df_normalized[banka].iloc[0] - 1) * 100
            xbank_perf = (df_normalized['XBANK'].iloc[-1] / df_normalized['XBANK'].iloc[0] - 1) * 100
            
            sonuclar[banka] = {
                'correlation': correlation,
                'p_value': p_value,
                'beta': beta,
                'perf': banka_perf
            }
            
    # XBANK için toplam performans
    xbank_perf = (df_normalized['XBANK'].iloc[-1] / df_normalized['XBANK'].iloc[0] - 1) * 100
    sonuclar['XBANK_perf'] = xbank_perf
    
    return df_normalized, df_returns, sonuclar

def cluster_banks(df_returns):
    """
    Kümeleme analizi ile benzer hareket eden bankaları gruplandır
    """
    # XBANK hariç analiz et
    df_cluster = df_returns.drop(columns=['XBANK'])
    
    # Boş değerleri temizle
    df_cluster = df_cluster.dropna(axis=1)
    
    # Kümeleme için veriyi hazırla
    features = pd.DataFrame(index=df_cluster.columns)
    
    # Volatilite (standart sapma)
    features['volatility'] = df_cluster.std()
    
    # XBANK ile korelasyon
    features['xbank_corr'] = [df_returns[col].corr(df_returns['XBANK']) for col in df_cluster.columns]
    
    # Beta
    features['beta'] = [
        np.cov(df_returns[col], df_returns['XBANK'])[0, 1] / np.var(df_returns['XBANK']) 
        for col in df_cluster.columns
    ]
    
    # Öznitelikleri ölçeklendir
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Optimal küme sayısını belirle (Elbow yöntemi)
    distortions = []
    K_range = range(1, min(8, len(df_cluster.columns)))
    for k in K_range:
        if k < len(df_cluster.columns):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(features_scaled)
            distortions.append(kmeans.inertia_)
    
    # Eğer yeterli veri yoksa
    if len(K_range) <= 1:
        optimal_k = 1
    else:
        # Basit elbow yöntemi - eğim değişimini kontrol et
        deltas = np.diff(distortions)
        if len(deltas) > 0:
            optimal_k = np.argmax(deltas) + 1
            # Minimum 2 küme olsun
            optimal_k = max(2, optimal_k)
            # Banka sayısının yarısından fazla olmasın
            optimal_k = min(optimal_k, len(df_cluster.columns) // 2)
        else:
            optimal_k = 1
    
    # KMeans uygula
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters_pred = kmeans.fit_predict(features_scaled)
    
    # Sonuçları organize et
    clusters = {}
    for i, bank in enumerate(df_cluster.columns):
        cluster_id = int(clusters_pred[i])
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(bank)
    
    # Cluster merkezlerini orijinal özelliklere dönüştür
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    center_df = pd.DataFrame(
        cluster_centers, 
        columns=features.columns,
        index=[f"Küme {i+1}" for i in range(optimal_k)]
    )
    
    # Her kümenin XBANK ile korelasyonu
    cluster_xbank_corr = {}
    for cluster_id, banks in clusters.items():
        # Her küme için ortalama getiriyi hesapla
        cluster_returns = df_cluster[banks].mean(axis=1)
        # XBANK ile korelasyonu
        corr = np.corrcoef(cluster_returns, df_returns['XBANK'])[0, 1]
        cluster_xbank_corr[cluster_id] = corr
    
    # Metrikleri hazırla
    metrics = {
        'optimal_k': optimal_k,
        'distortions': distortions,
        'cluster_centers': center_df,
        'cluster_xbank_corr': cluster_xbank_corr
    }
    
    return clusters, metrics

def detect_structural_breaks(series, min_size=30, n_bkps=5):
    """
    Zaman serisindeki yapısal kırılmaları tespit eder
    """
    # Eksik verileri temizle
    clean_series = series.dropna()
    
    # Ruptures algoritması için veriyi hazırla
    signal = clean_series.values.reshape(-1, 1)
    
    # Algoritma
    # PELT algoritması için penalty parametresi kullanmalıyız, n_bkps yerine
    algo = rpt.Pelt(model="l2", min_size=min_size).fit(signal)
    
    # İstenen kırılma sayısına göre uygun pen değerini hesapla
    # Pelt.predict() metodu pen parametresi alıyor, n_bkps değil
    # Manüel olarak pen değeri bulabiliriz
    pen_min = 1
    pen_max = 10000
    pen = 1000  # Başlangıç değeri
    
    # Binary search ile uygun pen değerini bul
    # Bu değer istediğimiz sayıda kırılma noktası verecek
    bkps = algo.predict(pen=pen)
    for _ in range(10):  # En fazla 10 deneme
        if len(bkps) - 1 < n_bkps:  # Kırılma sayısı azdır, pen azalt
            pen_max = pen
            pen = (pen_min + pen)/2
        elif len(bkps) - 1 > n_bkps:  # Kırılma sayısı fazladır, pen artır
            pen_min = pen
            pen = (pen + pen_max)/2
        else:
            break  # İstenen sayıda kırılma noktası bulundu
        bkps = algo.predict(pen=pen)
    
    # n_bkps+1 nokta varsa, ilk ve son noktalar başlangıç ve bitiş
    # eğer istediğimiz sayıyı bulamazsak, en azından yakın bir sonuç döndür
    
    # Orijinal indekse dönüştür
    bkps_dates = [clean_series.index[i-1] if i < len(clean_series) else clean_series.index[-1] for i in bkps if i > 0]
    
    return bkps_dates, algo

def analyze_structural_breaks(df, target_col="XBANK", n_bkps=5):
    """
    Yapısal kırılmaları analiz edip sonuçları döndürür
    """
    # Normalize edilmiş verileri kullan
    df_norm = df.copy()
    for col in df_norm.columns:
        df_norm[col] = df_norm[col] / df_norm[col].iloc[0] * 100
    
    # Yapısal kırılmaları tespit et
    break_dates, algo = detect_structural_breaks(
        df_norm[target_col], 
        min_size=30, 
        n_bkps=n_bkps
    )
    
    # Sonuçları hazırla
    break_results = {
        'target': target_col,
        'break_dates': break_dates,
        'algorithm': algo,
        'segments': []
    }
    
    # Her segment için istatistikleri hesapla
    segments = []
    prev_date = df_norm.index[0]
    
    for date in break_dates:
        if date <= prev_date:
            continue
            
        segment_data = df_norm.loc[prev_date:date]
        
        if len(segment_data) < 5:  # Çok kısa segmentleri atla
            continue
            
        # Segment başlangıç ve bitiş değerleri
        start_val = segment_data[target_col].iloc[0]
        end_val = segment_data[target_col].iloc[-1]
        
        # Segment için trend (yüzde değişim)
        trend_pct = ((end_val / start_val) - 1) * 100
        
        # Volatilite (standart sapma)
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
    
    # Son segment
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