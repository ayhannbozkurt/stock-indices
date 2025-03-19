# Banka Endeksi ve Hisseleri Analizi

Bu uygulama, BIST banka hisselerinin ve XBANK endeksinin performansını karşılaştıran ve analiz eden bir Streamlit uygulamasıdır.

## Özellikler

- XBANK endeksi ile banka hisselerinin performans karşılaştırması
- Normalize edilmiş fiyat hareketleri
- Korelasyon, Beta ve Performans analizleri
- Yuvarlanır korelasyon analizi
- Banka/XBANK performans oranı
- Beta-Performans ilişkisi grafiği
- Detaylı veri tablosu

## Kurulum

1. Repoyu klonlayın:
```
git clone https://github.com/yourusername/hisse-endeks-analizi.git
cd hisse-endeks-analizi
```

2. Gerekli paketleri yükleyin:
```
pip install -r requirements.txt
```

## Kullanım

Uygulamayı çalıştırmak için:
```
streamlit run main.py
```

## Proje Yapısı

- `main.py`: Streamlit arayüzü ve ana uygulama
- `data_utils.py`: Veri çekme ve analiz fonksiyonları
- `visualization.py`: Görselleştirme fonksiyonları
- `requirements.txt`: Gerekli paketler

## Veri Kaynağı

Veriler Yahoo Finance API'si üzerinden yfinance kütüphanesi kullanılarak çekilmektedir.

## Analizler

1. **Normalize Edilmiş Fiyat Karşılaştırması**: Tüm bankaların ve XBANK endeksinin normalize edilmiş fiyat hareketleri
2. **Korelasyon Analizi**: Banka hisselerinin XBANK ile korelasyonu
3. **Beta Analizi**: Banka hisselerinin XBANK ile beta katsayısı
4. **Performans Analizi**: Son 1 yıldaki performans karşılaştırması
5. **Yuvarlanır Korelasyon**: Belirli bir pencere boyutunda yuvarlanır korelasyon analizi
6. **Performans Oranı ve Farkı**: Bankaların XBANK'a göre göreceli performansı
7. **Beta-Performans İlişkisi**: Beta katsayısı ile performans arasındaki ilişkiyi gösteren dağılım grafiği

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. 