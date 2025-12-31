#  Yapay Zeka Destekli Meyve Sınıflandırma Uygulaması
**Yapay Zeka ve Bulut Bilişim Teknolojileri – Proje Ödevi**

---
##DataSet Olan Klasörümü Drive Linki:https://drive.google.com/file/d/10CanuV1FzCjav5SWWrjt1nMegxj9R1Gt/view?usp=drive_link

##  Proje Tanımı
Bu projede, yapay zeka destekli bir **görüntü sınıflandırma sistemi** geliştirilmiştir.  
Kullanıcı tarafından yüklenen meyve görselleri, sistem tarafından otomatik olarak ön işleme tabi tutulur ve **eğitilmiş bir derin öğrenme modeli** kullanılarak sınıflandırılır.  
Tahmin sonuçları, **kullanıcı dostu bir web arayüzü (Gradio)** üzerinden sunulmaktadır.

Bu çalışma, ders kapsamında verilen *“Yapay Zeka Destekli Görüntü Sınıflandırıcı”* proje ödevi doğrultusunda hazırlanmıştır.

---

##  Proje Amacı
- Görüntü sınıflandırma problemini çözmek  
- Derin öğrenme tabanlı modern bir mimari kullanmak  
- Görüntü ön işleme adımlarını uygulamak  
- Eğitilmiş modeli web tabanlı bir arayüz ile entegre etmek  
- Model performansını akademik metriklerle değerlendirmek  

---

##  Kullanılan Teknolojiler
- **Model:** Vision Transformer (ViT – `google/vit-base-patch16-224`)
- **Derin Öğrenme:** PyTorch, Hugging Face Transformers
- **Veri İşleme:** NumPy, Pandas
- **Model Değerlendirme:** Scikit-learn
- **Görselleştirme:** Matplotlib
- **Arayüz:** Gradio
- **Programlama Dili:** Python
- **Çalışma Ortamı:** CPU

---

## Veri Seti ve Sınıflar
Projede meyve sınıflandırma veri seti kullanılmıştır.  
Veri seti `ImageFolder` formatındadır ve aşağıdaki şekilde ayrılmıştır:

- **train** – Eğitim verisi  
- **validation** – Doğrulama verisi  
- **test** – Test verisi  

### Sınıflar (6 adet)
- freshapples  
- freshbanana  
- freshoranges  
- rottenapples  
- rottenbanana  
- rottenoranges  

---

## Ön İşleme Adımları
Model performansını artırmak amacıyla yüklenen görsellere aşağıdaki ön işlemler uygulanmaktadır:

- Yeniden boyutlandırma (**Resize**)  
- Merkezden kırpma (**Center Crop**)  
- RGB formatına dönüştürme  
- Normalizasyon  

Bu adımlar, modelin giriş boyutuna uygun ve kararlı tahminler üretmesini sağlamaktadır.

---

## Model Eğitimi
Model, önceden eğitilmiş **Vision Transformer (ViT)** mimarisi kullanılarak **transfer öğrenme** yaklaşımı ile eğitilmiştir.

### Eğitim Ayarları
- Optimizer: AdamW  
- Learning Rate: 2e-5  
- Epoch Sayısı: 2  
- Batch Size: 16  

---

## Model Performansı
Model başarımı aşağıdaki metrikler kullanılarak değerlendirilmiştir:

- **Accuracy (Doğruluk)**
- **Precision (Macro)**
- **Recall (Macro)**
- **F1-Score (Macro)**

### Eğitim Sonuçları
- Validation Accuracy: **%99+**
- Training ve Validation Loss değerleri epoch bazlı olarak düzenli şekilde azalmıştır.
- Overfitting problemi gözlemlenmemiştir.

Eğitim sürecine ait **Loss** ve **Accuracy** grafikleri (`loss.png`, `accuracy.png`) oluşturulmuştur.

---

## Confusion Matrix Analizi
Test verisi üzerinde oluşturulan **Confusion Matrix**, modelin sınıflandırma davranışını detaylı olarak analiz etmeyi sağlamaktadır.

Analiz sonucunda:
- Modelin meyve türlerini (**elma, muz, portakal**) büyük oranda doğru ayırt edebildiği görülmüştür.
- Diyagonal hücrelerin dolu olması, modelin rastgele tahmin yapmadığını göstermektedir.
- **Taze (fresh)** ve **çürük (rotten)** sınıfları arasında bazı karışmalar gözlemlenmiştir.

Bu durum, taze ve çürük meyvelerin renk, doku ve yüzey yapısı açısından görsel olarak benzer olmasından kaynaklanmaktadır. Görüntü tabanlı sınıflandırma problemlerinde bu tür karışmalar **beklenen ve kabul edilebilir** bir durumdur.

Oluşturulan Confusion Matrix görseli `confusion_matrix.png` dosyası olarak kaydedilmiştir.

---

## Kullanıcı Arayüzü (Gradio)
Geliştirilen web tabanlı arayüz sade ve kullanıcı dostu olacak şekilde tasarlanmıştır.

### Arayüz Özellikleri
- Görsel yükleme alanı  
- Ön işleme sonrası görsel önizleme  
- **Ön İşleme Uygula** butonu  
- **Tahmin Et** butonu  
- Tahmin edilen sınıf adı ve güven oranı (%)  
- **Top-5 tahmin listesi**  
- **Temizle** butonu  

Arayüz sayesinde kullanıcılar herhangi bir teknik bilgiye ihtiyaç duymadan modeli test edebilmektedir.

---

##  Kurulum ve Çalıştırma
Gerekli kütüphaneler aşağıdaki komut ile kurulabilir:

```bash
pip install -r requirements.txt
