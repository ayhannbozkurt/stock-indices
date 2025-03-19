import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import seaborn as sns

def plot_normalized_prices(df_normalized):
    """Tüm bankaların normalize edilmiş fiyat hareketleri"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Daha fazla renk kullanarak tüm bankaları ayırt etmeye çalış
    colors = list(mcolors.TABLEAU_COLORS) + list(mcolors.CSS4_COLORS)
    color_idx = 0
    
    for banka in df_normalized.columns:
        if banka != 'XBANK':
            ax.plot(df_normalized.index, df_normalized[banka], label=banka, alpha=0.7, color=colors[color_idx % len(colors)])
            color_idx += 1
    
    ax.plot(df_normalized.index, df_normalized['XBANK'], label='Borsa Endeksi', linewidth=3, color='black')
    ax.set_title('Normalize Edilmiş Fiyat Karşılaştırması (Başlangıç=100)')
    ax.legend(loc='upper left', fontsize='small')
    ax.grid(True)
    return fig

def plot_correlations(sonuclar):
    """Korelasyon karşılaştırması"""
    bankalar = [banka for banka in sonuclar.keys() if banka != 'XBANK_perf']
    korelasyonlar = [sonuclar[banka]['correlation'] for banka in bankalar]
    # Korelasyona göre sırala
    sort_indices = np.argsort(korelasyonlar)[::-1]  # Büyükten küçüğe
    korelasyonlar_sorted = [korelasyonlar[i] for i in sort_indices]
    bankalar_sorted = [bankalar[i] for i in sort_indices]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(bankalar_sorted, korelasyonlar_sorted)
    
    # Renklendirme
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.viridis(i/len(bars)))
    
    ax.set_title('Borsa Endeksi ile Korelasyon (Büyükten Küçüğe)')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, axis='y')
    return fig, bankalar_sorted

def plot_betas(sonuclar):
    """Beta karşılaştırması"""
    bankalar = [banka for banka in sonuclar.keys() if banka != 'XBANK_perf']
    betalar = [sonuclar[banka]['beta'] for banka in bankalar]
    # Beta'ya göre sırala
    sort_indices = np.argsort(betalar)[::-1]  # Büyükten küçüğe
    betalar_sorted = [betalar[i] for i in sort_indices]
    bankalar_beta_sorted = [bankalar[i] for i in sort_indices]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(bankalar_beta_sorted, betalar_sorted)
    
    # Renklendirme
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.plasma(i/len(bars)))
    
    ax.set_title('Borsa Endeksi ile Beta (Büyükten Küçüğe)')
    ax.axhline(y=1, color='r', linestyle='--', label='Beta=1')
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, axis='y')
    ax.legend()
    return fig, bankalar_beta_sorted

def plot_performance(sonuclar):
    """Son 1 yıllık performans"""
    bankalar = [banka for banka in sonuclar.keys() if banka != 'XBANK_perf']
    performanslar = [sonuclar[banka]['perf'] for banka in bankalar]
    # Performansa göre sırala
    sort_indices = np.argsort(performanslar)[::-1]  # Büyükten küçüğe
    performanslar_sorted = [performanslar[i] for i in sort_indices]
    bankalar_perf_sorted = [bankalar[i] for i in sort_indices]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(bankalar_perf_sorted, performanslar_sorted)
    
    # Renklendirme - performans değerlerine göre
    norm = plt.Normalize(min(performanslar_sorted), max(performanslar_sorted))
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.coolwarm(norm(performanslar_sorted[i])))
    
    xbank_perf = sonuclar['XBANK_perf']
    ax.axhline(y=xbank_perf, color='r', linestyle='--', label=f'Borsa Endeksi: {xbank_perf:.2f}%')
    ax.set_title('Son 1 Yıllık Performans (%)')
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, axis='y')
    ax.legend()
    return fig, bankalar_perf_sorted

def plot_rolling_correlation(df_returns, bankalar, window_size=30):
    """Seçilen bankaların XBANK ile korelasyon rollinglerini çiz"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Renk paleti oluştur
    colors = plt.cm.rainbow(np.linspace(0, 1, len(bankalar)))
    
    for i, banka in enumerate(bankalar):
        rolling_corr = df_returns[banka].rolling(window=window_size).corr(df_returns['XBANK'])
        ax.plot(rolling_corr.index, rolling_corr, label=banka, alpha=0.7, linewidth=1.5, color=colors[i])
    
    ax.set_title(f'{window_size} Günlük Yuvarlanır Korelasyon (Borsa Endeksi ile)')
    # Grafiği daha okunabilir yapmak için lejantı dışarı çıkar
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax.grid(True)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    return fig

def plot_relative_performance(df_normalized, bankalar):
    """Bankaların XBANK'a göre performansı"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Renk paleti oluştur
    colors = plt.cm.viridis(np.linspace(0, 1, len(bankalar)))
    
    for i, banka in enumerate(bankalar):
        relative_perf = df_normalized[banka] / df_normalized['XBANK'] * 100
        ax.plot(df_normalized.index, relative_perf, label=banka, alpha=0.7, linewidth=1.5, color=colors[i])
    
    ax.set_title('Banka/Borsa Endeksi Performans Oranı')
    ax.axhline(y=100, color='r', linestyle='-', alpha=0.3, label='Borsa Endeksi')
    # Grafiği daha okunabilir yapmak için lejantı dışarı çıkar
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax.grid(True)
    plt.tight_layout()
    return fig

def plot_cumulative_diff(df_normalized, bankalar):
    """Kümülatif performans farkı"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Renk paleti oluştur
    colors = plt.cm.plasma(np.linspace(0, 1, len(bankalar)))
    
    for i, banka in enumerate(bankalar):
        cumulative_diff = (df_normalized[banka] - df_normalized['XBANK'])
        ax.plot(df_normalized.index, cumulative_diff, label=banka, alpha=0.7, linewidth=1.5, color=colors[i])
    
    ax.set_title('Banka - Borsa Endeksi Normalize Fiyat Farkı')
    ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    # Grafiği daha okunabilir yapmak için lejantı dışarı çıkar
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax.grid(True)
    plt.tight_layout()
    return fig

def plot_beta_performance(sonuclar):
    """Endeks-Beta-Performans ilişkisi"""
    bankalar = [banka for banka in sonuclar.keys() if banka != 'XBANK_perf']
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Korelasyon değerlerine göre renklendirme
    korelasyonlar = [sonuclar[banka]['correlation'] for banka in bankalar]
    scatter = ax.scatter(
        [sonuclar[banka]['beta'] for banka in bankalar], 
        [sonuclar[banka]['perf'] for banka in bankalar], 
        alpha=0.7, s=100, c=korelasyonlar, cmap='viridis'
    )
    
    # Renk çubuğu ekle
    cbar = plt.colorbar(scatter)
    cbar.set_label('Korelasyon')

    # Noktalara banka isimlerini ekle
    for banka in bankalar:
        ax.annotate(banka, 
                   (sonuclar[banka]['beta'], sonuclar[banka]['perf']),
                   xytext=(5, 5), textcoords='offset points')

    ax.set_title('Beta - Performans İlişkisi')
    ax.set_xlabel('Beta Katsayısı')
    ax.set_ylabel('Son 1 Yıllık Performans (%)')
    ax.axhline(y=sonuclar['XBANK_perf'], color='r', linestyle='--', 
               label=f'Borsa Endeksi: {sonuclar["XBANK_perf"]:.2f}%')
    ax.axvline(x=1, color='g', linestyle='--', label='Beta = 1')
    ax.grid(True)
    ax.legend()
    return fig

def plot_structural_breaks(df_normalized, break_results):
    """Yapısal kırılmaları grafikle görselleştir"""
    target_col = break_results['target']
    break_dates = break_results['break_dates']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Hedef seriyi çiz
    ax.plot(df_normalized.index, df_normalized[target_col], label=target_col if target_col != 'XBANK' else 'Borsa Endeksi', linewidth=2, color='blue')
    
    # Kırılma noktalarını işaretle
    for date in break_dates:
        if date in df_normalized.index:
            ax.axvline(x=date, color='red', linestyle='--', alpha=0.7)
            # Tarih etiketini ekle
            ax.text(date, df_normalized[target_col].max() * 1.05, 
                   date.strftime('%Y-%m-%d'), 
                   rotation=45, ha='right')
    
    # Segmentleri renklendir
    segments = break_results['segments']
    colors = plt.cm.tab10(np.linspace(0, 1, len(segments)))
    
    for i, segment in enumerate(segments):
        start_date = segment['start_date']
        end_date = segment['end_date']
        trend = segment['trend_pct']
        
        # Segment aralığını belirle
        segment_idx = (df_normalized.index >= start_date) & (df_normalized.index <= end_date)
        
        # Segment üzerine şeffaf renk ekle
        ax.fill_between(df_normalized.index[segment_idx], 
                       0, df_normalized[target_col].max() * 1.1,
                       alpha=0.1, color=colors[i % len(colors)])
        
        # Trend yönünü göster
        mid_point = df_normalized.index[segment_idx].mean()
        trend_text = f"{trend:.1f}%"
        y_pos = df_normalized[target_col].min() * 0.9
        
        # Trendin işaretine göre renk belirle
        text_color = 'green' if trend > 0 else 'red'
        ax.text(mid_point, y_pos, trend_text, color=text_color, 
               ha='center', va='bottom', fontweight='bold')
    
    title_text = 'Borsa Endeksi Yapısal Kırılma Analizi' if target_col == 'XBANK' else f'{target_col} Yapısal Kırılma Analizi'
    ax.set_title(title_text)
    ax.set_xlabel('Tarih')
    ax.set_ylabel('Normalize Değer (100)')
    ax.grid(True)
    plt.tight_layout()
    
    return fig

def plot_cluster_analysis(df_normalized, clusters, metrics):
    """Kümeleme analizini görselleştir"""
    # 1. Küme üyelerini göster
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    # Çubuk grafik için veri hazırla
    cluster_counts = {f"Küme {i+1}": len(banks) for i, banks in clusters.items()}
    x = list(cluster_counts.keys())
    y = list(cluster_counts.values())
    
    # Çubukları çiz
    bars = ax1.bar(x, y)
    
    # Çubukların üzerine sayıları yaz
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f"{height}", ha='center', va='bottom')
    
    ax1.set_title("Kümeleme Analizi - Küme Büyüklükleri")
    ax1.set_xlabel("Küme")
    ax1.set_ylabel("Banka Sayısı")
    
    # 2. PCA ile 2D görselleştirme - Korelasyon matrisini kullan
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    # Küme renkleri
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))
    
    # Her banka için grafik çiz
    for cluster_id, banks in clusters.items():
        # Her küme için, kümedeki bankaların normalize edilmiş fiyatlarının ortalamasını al
        cluster_data = df_normalized[banks].mean(axis=1)
        ax2.plot(df_normalized.index, cluster_data, label=f"Küme {cluster_id+1}", 
                linewidth=2, color=cluster_colors[cluster_id % len(cluster_colors)])
    
    # XBANK de ekle
    ax2.plot(df_normalized.index, df_normalized['XBANK'], label='Borsa Endeksi', 
            linewidth=3, color='black', linestyle='--')
    
    ax2.set_title("Banka Kümeleri - Ortalama Fiyat Hareketleri")
    ax2.set_xlabel("Tarih")
    ax2.set_ylabel("Normalize Değer (100)")
    ax2.legend()
    ax2.grid(True)
    
    # 3. Kümelerin özelliklerini gösteren ısı haritası
    cluster_centers = metrics['cluster_centers']
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    
    # Isı haritası oluştur
    sns.heatmap(cluster_centers, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax3)
    ax3.set_title("Küme Merkezleri - Özellikler")
    
    # 4. Korelasyon matrisi ve kümeleme
    fig4, ax4 = plt.subplots(figsize=(12, 10))
    
    # XBANK hariç tüm bankaların günlük getirilerini al
    corr_matrix = df_normalized.drop(columns=['XBANK']).pct_change().dropna().corr()
    
    # Kümelere göre sırala
    ordered_banks = []
    for cluster_id in sorted(clusters.keys()):
        ordered_banks.extend(clusters[cluster_id])
    
    # Sıralanmış korelasyon matrisi
    corr_ordered = corr_matrix.loc[ordered_banks, ordered_banks]
    
    # Isı haritası
    mask = np.zeros_like(corr_ordered)
    mask[np.triu_indices_from(mask)] = True
    
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr_ordered, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
               square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax4)
    
    # Küme sınırlarını göster
    current_idx = 0
    for cluster_id in sorted(clusters.keys()):
        cluster_size = len(clusters[cluster_id])
        # Dikdörtgen çiz
        rect = plt.Rectangle((current_idx, current_idx), cluster_size, cluster_size, 
                          fill=False, linewidth=2, edgecolor='red')
        ax4.add_patch(rect)
        current_idx += cluster_size
    
    ax4.set_title("Banka Getirilerinin Korelasyon Matrisi (Kümelere Göre Sıralanmış)")
    
    return [fig1, fig2, fig3, fig4]

def plot_cluster_members(clusters, selected_cluster=None):
    """Küme üyelerini tablo olarak görselleştir"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Tablo verilerini hazırla
    if selected_cluster is not None and selected_cluster in clusters:
        # Sadece seçilen kümeyi göster
        table_data = [clusters[selected_cluster]]
        row_labels = [f"Küme {selected_cluster+1}"]
    else:
        # Tüm kümeleri göster
        table_data = [clusters[i] for i in sorted(clusters.keys())]
        row_labels = [f"Küme {i+1}" for i in sorted(clusters.keys())]
    
    # En uzun listeyi bul ve diğerlerini boşluklarla doldur
    max_len = max(len(cluster) for cluster in table_data)
    padded_data = [cluster + [''] * (max_len - len(cluster)) for cluster in table_data]
    
    # Tablo oluştur
    table = ax.table(cellText=padded_data, rowLabels=row_labels,
                    loc='center', cellLoc='center')
    
    # Tablo formatı
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Başlık
    if selected_cluster is not None:
        ax.set_title(f"Küme {selected_cluster+1} Bankaları", fontsize=14)
    else:
        ax.set_title("Banka Kümeleri", fontsize=14)
    
    return fig

def plot_performance_heatmap(df_returns, bankalar):
    """
    Bankaların aylık performansını gösteren ısı haritası.
    Renk skalası: Kırmızı (negatif) - Beyaz (nötr) - Yeşil (pozitif)
    """
    # Aylık getiri hesapla
    monthly_returns = df_returns.copy()
    monthly_returns.index = pd.to_datetime(monthly_returns.index)
    monthly_returns = monthly_returns.resample('M').apply(
        lambda x: ((1 + x).prod() - 1) * 100  # Aylık %'de getiri
    )
    
    # Sadece bankaları seç
    monthly_bank_returns = monthly_returns[bankalar].copy()
    # Ayları formatla
    monthly_bank_returns.index = monthly_bank_returns.index.strftime('%Y-%m')
    
    # Transpoze et (bankalar satır, aylar sütun olsun)
    data = monthly_bank_returns.transpose()
    
    # Görselleştirme
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Isı haritası
    cmap = sns.diverging_palette(10, 130, as_cmap=True)  # Kırmızı-Yeşil
    heatmap = sns.heatmap(data, cmap=cmap, center=0, annot=True, fmt=".1f", 
                          linewidths=.5, ax=ax, cbar_kws={'label': 'Aylık Getiri (%)'})
    
    # Başlık ve eksen etiketleri
    ax.set_title('Bankaların Aylık Performansı (%)', fontsize=16)
    ax.set_xlabel('Tarih', fontsize=12)
    ax.set_ylabel('Banka', fontsize=12)
    
    # X ekseni etiketlerini döndür
    plt.xticks(rotation=45, ha='right')
    
    return fig 