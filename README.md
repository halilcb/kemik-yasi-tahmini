# Kemik Yaşı Tahmin Projesi

Bu proje, çocukların el röntgeni görüntülerini kullanarak kemik yaşını tahmin etmek için derin öğrenme tekniklerini uygulamaktadır. Projede transfer öğrenme yöntemiyle **Xception** modeli kullanılarak yüksek doğrulukta tahminler elde edilmesi hedeflenmiştir.



## Veri Seti

Bu projede, **RSNA 2017 Bone Age Challenge** yarışması kapsamında sağlanan el röntgeni görüntüleri kullanılmıştır. Veri seti, Kaggle üzerinden temin edilmiştir.

Veri seti şu öğeleri içermektedir:
- **El Röntgeni Görüntüleri**: Dijital ve taranmış formatlarda.
- **CSV Dosyası**: Her görüntü için aşağıdaki bilgiler yer almaktadır:
  - **Bone Age**: Tahmin edilmesi gereken kemik yaşı (ay cinsinden).
  - **Gender**: Ek özellik olarak kullanılan cinsiyet bilgisi.

**Veri Kaynağı:** RSNA (Radiological Society of North America), Stanford Üniversitesi, Colorado Üniversitesi ve UCLA katkılarıyla oluşturulmuştur.


## Proje Aşamaları

### 1. Veri Yükleme ve Ön İşleme

- Görüntü dosyaları ve etiketler pandas ile yüklendi.
- `id` sütunu kullanılarak her görüntü için dosya yolları oluşturuldu.
- Cinsiyet verisi ikili kodlama ile `0` (kadın) ve `1` (erkek) olarak etiketlendi.
- Kemik yaşı normalize edilerek yeni bir sütun oluşturuldu.

### 2. Veri Artırma (Data Augmentation)

- **Dönme (Rotation):** ±30 derece
- **Yakınlaştırma (Zoom):** 0.1 oranında
- **Parlaklık Ayarı (Brightness Range):** 0.8 - 1.2 aralığında
- **Yatay Çevirme (Horizontal Flip)**
- **Kaydırma (Shift):** Yatay ve dikey yönde %10 oranında

Bu adımlar modeli aşırı öğrenmeden (overfitting) korumak için uygulanmıştır.

### 3. Veri Bölme

- Veriler, eğitim, doğrulama ve test setlerine ayrıldı:
  - Eğitim seti (%80)
  - Doğrulama seti (%10)
  - Test seti (%10)

### 4. Model Kurulumu

- **Transfer Learning:** Önceden ImageNet veri setiyle eğitilmiş **Xception** modeli kullanıldı.
- Modelin eğitilebilir (trainable) olması sağlanarak ince ayar (fine-tuning) yapıldı.
- Özel katmanlar eklendi:
  - `GlobalMaxPooling2D`
  - `Flatten`
  - `Dense` (ReLU aktivasyon fonksiyonu ile)
  - Son katmanda `linear` aktivasyon ile kemik yaşı tahmini yapıldı.

### 5. Model Eğitimi

- **Optimizasyon:** `Adam` optimizasyon algoritması (`learning_rate=0.0005`)
- **Kayıp Fonksiyonu:** Ortalama Kare Hatası (MSE)
- **Erken Durdurma (Early Stopping):** Modelin aşırı öğrenmesini önlemek için kullanıldı.
- Eğitim, 3 epoch boyunca gerçekleştirildi.

### 6. Model Değerlendirme

Model, test verisi üzerinde değerlendirildi:
- **MSE (Ortalama Kare Hata):** Modelin tahmin başarımını ölçmek için kullanıldı.
- **Görselleştirme:** Test setinden rastgele seçilen görüntüler üzerinde gerçek ve tahmin edilen kemik yaşları karşılaştırıldı.



## Model Performansı

Eğitim sonucunda aşağıdaki sonuçlar elde edilmiştir:
- **Validation Loss:** ~202.41 (MSE)
- **Test Performansı:** Tahmin edilen yaşlar gerçek değerlere oldukça yakındır.


