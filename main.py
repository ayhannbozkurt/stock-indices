import streamlit as st
from datetime import timedelta
import data_utils
import visualization

# Add page title and description
st.set_page_config(
    page_title="Banking Index Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"  # Keep sidebar expanded initially
)

# Add CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 1.5rem;
    }
    .positive-delta {
        color: green !important;
    }
    .negative-delta {
        color: red !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>Banking Index and Stocks Analysis</h1>", unsafe_allow_html=True)
st.write("This application compares the performance of the Banking Index and bank stocks.")

# Veri işleme
@st.cache_data(ttl=timedelta(hours=6))
def load_data(days=365):
    # Verileri çek ve analiz et
    df = data_utils.fetch_bank_data(days)
    df_normalized, df_returns, sonuclar = data_utils.analyze_bank_data(df)
    
    # Kümeleme analizi
    clusters, cluster_metrics = data_utils.cluster_banks(df_returns)
    
    # Yapısal kırılma analizi
    structural_breaks_xbank = data_utils.analyze_structural_breaks(df, "XBANK", n_bkps=5)
    
    return df, df_normalized, df_returns, sonuclar, clusters, cluster_metrics, structural_breaks_xbank

# Ana içerik
with st.spinner("Veriler indiriliyor ve analiz ediliyor..."):
    df, df_normalized, df_returns, sonuclar, clusters, cluster_metrics, structural_breaks_xbank = load_data()
# Bankacılık sektörüne genel bakış bölümü - kart stilinde metrikler
st.markdown("<h2 class='sub-header'>Banking Sector Overview</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    delta_class = "positive-delta" if sonuclar['XBANK_perf'] > 0 else "negative-delta"
    st.metric(
        "Banking Index Performance (Last 1 Year)", 
        f"{sonuclar['XBANK_perf']:.2f}%", 
        delta=f"{sonuclar['XBANK_perf']:.2f}%",
        delta_color="normal"
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    # En yüksek performansa sahip banka
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    bankalar = [banka for banka in sonuclar.keys() if banka != 'XBANK_perf']
    # Hata kontrolü: bankalar listesi boş olabilir
    if bankalar:
        en_iyi_banka = max(bankalar, key=lambda x: sonuclar[x]['perf'])
        en_iyi_perf = sonuclar[en_iyi_banka]['perf']
        st.metric(
            f"Best Performance: {en_iyi_banka}", 
            f"{en_iyi_perf:.2f}%", 
            delta=f"{en_iyi_perf - sonuclar['XBANK_perf']:.2f}%"
        )
    else:
        st.warning("Insufficient bank data found.")
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    # En düşük performansa sahip banka
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    if bankalar:
        en_kotu_banka = min(bankalar, key=lambda x: sonuclar[x]['perf'])
        en_kotu_perf = sonuclar[en_kotu_banka]['perf']
        st.metric(
            f"Worst Performance: {en_kotu_banka}", 
            f"{en_kotu_perf:.2f}%", 
            delta=f"{en_kotu_perf - sonuclar['XBANK_perf']:.2f}%"
        )
    else:
        st.warning("Insufficient bank data found.")
    st.markdown("</div>", unsafe_allow_html=True)

# Ana sekmeler - daha belirgin sekme stilleri
st.markdown("<h2 class='sub-header'>Analyses</h2>", unsafe_allow_html=True)
main_tabs = st.tabs(["📈 Basic Analyses", "🔍 Detailed Analyses"])

# Temel Analizler Sekmesi
with main_tabs[0]:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Normalized Price Movements")
        fig = visualization.plot_normalized_prices(df_normalized)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Banking Index Correlation")
        fig, bankalar_sorted = visualization.plot_correlations(sonuclar)
        st.pyplot(fig)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Banking Index Beta")
        fig, bankalar_beta_sorted = visualization.plot_betas(sonuclar)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Last 1 Year Performance")
        fig, bankalar_perf_sorted = visualization.plot_performance(sonuclar)
        st.pyplot(fig)

# Detaylı Analizler Sekmesi
with main_tabs[1]:
    detail_tabs = st.tabs([
        "Structural Break Analysis", 
        "Cluster Analysis", 
        "Comparative Analyses",
        "Performance Map",
        "Raw Data"
    ])
    
    # Yapısal Kırılma Analizi
    with detail_tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Banking Index Structural Break Analysis")
            
            # Kırılma figürü
            fig = visualization.plot_structural_breaks(df_normalized, structural_breaks_xbank)
            st.pyplot(fig)
        
        with col2:
            # Diğer banka analizlerini oluştur
            st.subheader("Structural Break Analysis for Another Bank")
            
            if bankalar:
                selected_bank = st.selectbox("Select a Bank", bankalar)
                
                if selected_bank:
                    # Seçilen banka için yapısal kırılma analizi
                    with st.spinner(f"Performing structural break analysis for {selected_bank}..."):
                        bank_break_results = data_utils.analyze_structural_breaks(
                            df, selected_bank, n_bkps=5
                        )
                        
                        # Analiz figürü
                        fig = visualization.plot_structural_breaks(df_normalized, bank_break_results)
                        st.pyplot(fig)
    
    # Kümeleme Analizi
    with detail_tabs[1]:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Bank Clusters")
            
            # Küme üyeleri
            fig = visualization.plot_cluster_members(clusters)
            st.pyplot(fig)
            
            # Küme merkezleri
            st.subheader("Cluster Properties")
            st.dataframe(cluster_metrics['cluster_centers'].style.format('{:.4f}'))
        
        with col2:
            st.subheader("Cluster Analysis Results")
            
            # Küme analizleri
            cluster_figs = visualization.plot_cluster_analysis(df_normalized, clusters, cluster_metrics)
            
            # İlk grafiği göster
            st.pyplot(cluster_figs[1])
            
            # Korelasyon matrisi ısı haritası
            st.subheader("Correlation Matrix (By Clusters)")
            st.write("This heatmap shows the correlation between banks. Red frames indicate clusters. Blue tones indicate positive correlation, red tones indicate negative correlation.")
            st.pyplot(cluster_figs[3])
    
    # Karşılaştırmalı Analizler
    with detail_tabs[2]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Rolling Correlation Analysis")
            window_size = st.slider("Rolling Correlation Window Size (Days)", 15, 90, 30)
            
            # Bankalar listesi boş değilse grafiği çiz
            if bankalar:
                # Banka seçimi
                selected_banks = st.multiselect(
                    "Select Banks (max. 5)", 
                    options=bankalar,
                    default=bankalar_sorted[:min(5, len(bankalar_sorted))]
                )
                
                if selected_banks:
                    # En fazla 5 banka göster
                    selected_banks = selected_banks[:5]
                    
                    # Grafiği çiz
                    fig = visualization.plot_rolling_correlation(df_returns, selected_banks, window_size)
                    st.pyplot(fig)
        
        with col2:
            st.subheader("Bank/Banking Index Performance Ratio")
            
            if bankalar:
                # Banka seçimi
                selected_banks = st.multiselect(
                    "Select Banks (max. 5)", 
                    options=bankalar,
                    default=bankalar_beta_sorted[:min(5, len(bankalar_beta_sorted))],
                    key="perf_ratio_select"
                )
                
                if selected_banks:
                    # En fazla 5 banka göster
                    selected_banks = selected_banks[:5]
                    
                    # Grafiği çiz
                    fig = visualization.plot_relative_performance(df_normalized, selected_banks)
                    st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Bank - Banking Index Normalized Price Difference")
            
            if bankalar:
                # Banka seçimi
                selected_banks = st.multiselect(
                    "Select Banks (max. 5)", 
                    options=bankalar,
                    default=bankalar_perf_sorted[:min(5, len(bankalar_perf_sorted))],
                    key="norm_diff_select"
                )
                
                if selected_banks:
                    # En fazla 5 banka göster
                    selected_banks = selected_banks[:5]
                    
                    # Grafiği çiz
                    fig = visualization.plot_cumulative_diff(df_normalized, selected_banks)
                    st.pyplot(fig)
        
        with col2:
            st.subheader("Beta - Performance Relationship")
            fig = visualization.plot_beta_performance(sonuclar)
            st.pyplot(fig)
    
    # Performans Haritası
    with detail_tabs[3]:
        st.subheader("Performance Map")
        
        st.write("A heatmap showing the monthly performance of banks.")
        if bankalar:
            # En fazla 10 banka göster
            selected_banks = st.multiselect(
                "Select Banks (max. 10)", 
                options=bankalar,
                default=bankalar_perf_sorted[:min(8, len(bankalar_perf_sorted))],
                key="heatmap_select"
            )
            
            if selected_banks:
                selected_banks = selected_banks[:10]
                fig = visualization.plot_performance_heatmap(df_returns, selected_banks)
                st.pyplot(fig)
    
    # Ham Veri
    with detail_tabs[4]:
        st.subheader("Raw Price Data")
        st.dataframe(df)
