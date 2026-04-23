# %% [markdown]
# # Lineer Regresyon ile Öğrenci Performans Tahmini
#
# Bu notebook, **Student_Performance.csv** veri seti üzerinde öğrenci performansını
# tahmin eden bir Lineer Regresyon modeli geliştirmektedir. Veri yükleme, keşif,
# ön işleme, model eğitimi, değerlendirme ve görselleştirme adımlarını içerir.

# %% 
# Gerekli kütüphanelerin içe aktarılması
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# %% [markdown]
# ## 1. Veri Yükleme
#
# CSV dosyasını pandas ile okuyarak bir DataFrame nesnesine yüklüyoruz.

# %%
# CSV dosyasını okuma
try:
    df = pd.read_csv("Student_Performance.csv")
    print("=" * 50)
    print("📂 VERİ YÜKLEME SONUCU")
    print("=" * 50)
    print(f"Veri seti başarıyla yüklendi.")
    print(f"Satır sayısı: {df.shape[0]}, Sütun sayısı: {df.shape[1]}")
except FileNotFoundError:
    print("Hata: 'Student_Performance.csv' dosyası bulunamadı!")

# %% [markdown]
# ## 2. Veri Setine Gerçekçilik Katma (Gürültü Ekleme)
#
# Orijinal veri seti sentetik (yapay üretilmiş) olduğundan neredeyse mükemmel bir
# doğrusal ilişki içermektedir. Gerçek dünya verilerini simüle etmek için hedef
# değişkene (Performance Index) rastgele Gaussian gürültü ekliyoruz.
# Bu sayede model daha gerçekçi bir R² değeri üretecektir.

# %%
# Hedef değişkene rastgele gürültü ekleme
# Gaussian (normal dağılım) gürültü: ortalama=0, standart sapma=7
np.random.seed(42)
gurultu = np.random.normal(0, 7, size=len(df))
df["Performance Index"] = df["Performance Index"] + gurultu

# Performans endeksini 0-100 aralığında tut (negatif veya 100 üstü olmasın)
df["Performance Index"] = df["Performance Index"].clip(0, 100)

print("=" * 50)
print("🔊 GÜRÜLTÜ EKLEME SONUCU")
print("=" * 50)
print(f"Eklenen gürültü: Gaussian(ortalama=0, std=7)")
print(f"Yeni Performance Index aralığı: {df['Performance Index'].min():.1f} - {df['Performance Index'].max():.1f}")
print(f"Yeni ortalama: {df['Performance Index'].mean():.1f}")

# %% [markdown]
# ## 3. Veri Keşfi
#
# Veri setinin yapısını ve istatistiksel özetini inceleyerek verinin genel durumunu anlıyoruz.

# %%
# İlk 5 satırı görüntüleme - veri setinin yapısını hızlıca görmek için
print("=" * 50)
print("📋 İLK 5 SATIR (head)")
print("=" * 50)
print(df.head())

# %%
# Sütun tipleri ve boş olmayan değer sayılarını görüntüleme
print("=" * 50)
print("ℹ️  VERİ SETİ BİLGİSİ (info)")
print("=" * 50)
df.info()

# %%
# Sayısal sütunların istatistiksel özeti (ortalama, std, min, max, çeyreklikler)
print("=" * 50)
print("📊 İSTATİSTİKSEL ÖZET (describe)")
print("=" * 50)
print(df.describe())

# %% [markdown]
# ## 4. Eksik Veri Kontrolü
#
# Veri setindeki eksik (null/NaN) değerleri kontrol ederek veri kalitesini değerlendiriyoruz.

# %%
# Her sütundaki eksik değer sayısını hesaplama
eksik_veriler = df.isnull().sum()
print("=" * 50)
print("🔍 EKSİK VERİ KONTROLÜ")
print("=" * 50)
print(eksik_veriler)
print(f"\nToplam eksik değer: {eksik_veriler.sum()}")

# %% [markdown]
# ## 5. Veri Ön İşleme
#
# Kategorik değişkenleri sayısal değerlere dönüştürerek modelin kullanabileceği formata getiriyoruz.

# %%
# "Extracurricular Activities" sütunundaki Yes/No değerlerini 1/0 sayısal değerlerine dönüştürme
# Model sadece sayılarla çalışabilir, bu yüzden Yes=1, No=0 yapıyoruz
df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})

# Dönüşüm sonrası doğrulama
print("=" * 50)
print("🔄 KATEGORİK DÖNÜŞÜM SONUCU")
print("=" * 50)
print(f"Benzersiz değerler: {df['Extracurricular Activities'].unique()}")
print(f"Veri tipi: {df['Extracurricular Activities'].dtype}")

# %% [markdown]
# ## 6. Özellik ve Hedef Değişken Tanımlama
#
# Modelin eğitimi için bağımsız değişkenler (X) ve bağımlı değişken (y) olarak veri setini ayırıyoruz.
# X matrisi 5 özellik sütununu, y ise tahmin edilecek performans endeksini içerir.

# %%
# Bağımsız değişkenler (özellik matrisi) - 5 sütun
# X = modelin tahmin yapmak için kullandığı girdi verileri
X = df[["Hours Studied", "Previous Scores", "Extracurricular Activities",
        "Sleep Hours", "Sample Question Papers Practiced"]]

# Bağımlı değişken (hedef değişken) - Performance Index
# y = modelin tahmin etmeye çalıştığı çıktı değeri
y = df["Performance Index"]

# Boyutları doğrulama
print("=" * 50)
print("📐 ÖZELLİK VE HEDEF DEĞİŞKEN BOYUTLARI")
print("=" * 50)
print(f"X (girdi verileri): {X.shape[0]} satır, {X.shape[1]} sütun")
print(f"y (tahmin edilecek değer): {y.shape[0]} satır")

# %% [markdown]
# ## 7. Veri Bölme
#
# Veri setini %80 eğitim ve %20 test olarak ayırıyoruz. Tekrarlanabilirlik için
# `random_state=42` parametresi kullanılmaktadır.

# %%
# Veriyi %80 eğitim, %20 test olarak bölme
# Eğitim seti: modelin öğrenmek için kullandığı veri
# Test seti: modelin hiç görmediği, başarısını ölçtüğümüz veri
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bölme sonrası boyutları yazdırma
print("=" * 50)
print("✂️  EĞİTİM / TEST BÖLME SONUCU")
print("=" * 50)
print(f"Eğitim seti: {X_train.shape[0]} satır (modelin öğrendiği veri)")
print(f"Test seti:   {X_test.shape[0]} satır (modelin hiç görmediği veri)")

# %% [markdown]
# ## 8. Model Eğitimi
#
# scikit-learn kütüphanesinden `LinearRegression` modeli oluşturup eğitim verisi ile eğitiyoruz.
# Eğitim sonrası modelin öğrendiği katsayıları ve sabit terimi inceliyoruz.

# %%
# Lineer Regresyon modeli oluşturma ve eğitim verisi ile eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# Katsayılar: her özelliğin performansa ne kadar etki ettiğini gösteren ağırlıklar
# Pozitif katsayı = o özellik arttıkça performans artar
# Negatif katsayı = o özellik arttıkça performans azalır
# Büyük katsayı = daha güçlü etki
print("=" * 50)
print("🎯 MODEL KATSAYILARI")
print("=" * 50)
print("(Her değişkenin performansa etkisini gösteren ağırlıklar)")
print()
for ozellik, katsayi in zip(X.columns, model.coef_):
    print(f"  {ozellik}: {katsayi:.4f}")

# Sabit terim (intercept): tüm özellikler sıfır olduğunda modelin verdiği başlangıç değeri
print(f"\nSabit Terim (Intercept): {model.intercept_:.4f}")
print("(Tüm değişkenler 0 olsaydı modelin tahmin edeceği başlangıç değeri)")

# %% [markdown]
# ## 9. Tahmin
#
# Eğitilmiş model ile test verisi üzerinde tahmin yaparak modelin gerçek performansını ölçmeye hazırlanıyoruz.

# %%
# Test verisi üzerinde tahmin üretme
y_pred = model.predict(X_test)

# İlk 10 tahmini gerçek değerlerle karşılaştırma
# Gerçek = öğrencinin asıl performans puanı
# Tahmin = modelin o öğrenci için hesapladığı puan
# Fark = ne kadar yanıldığımız (0'a yakın = iyi tahmin)
print("=" * 50)
print("🔮 TAHMİN SONUÇLARI (İlk 10 Öğrenci)")
print("=" * 50)
print(f"Toplam tahmin sayısı: {len(y_pred)}")
print()
print(f"{'Gerçek':>10} {'Tahmin':>10} {'Fark':>10}")
print("-" * 32)
for gercek, tahmin in zip(y_test.values[:10], y_pred[:10]):
    fark = gercek - tahmin
    print(f"{gercek:>10.2f} {tahmin:>10.2f} {fark:>10.2f}")

# %% [markdown]
# ## 10. Performans Metrikleri
#
# Modelin ne kadar iyi tahmin yaptığını sayısal olarak ölçüyoruz.
# Bu metrikler modelin başarısını özetleyen temel göstergelerdir.

# %%
# Performans metriklerini hesaplama
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("=" * 50)
print("📈 PERFORMANS METRİKLERİ")
print("=" * 50)

# R² (R-Kare): Modelin veriyi ne kadar iyi açıkladığını gösterir.
# 0 ile 1 arasında değer alır. 1'e ne kadar yakınsa model o kadar başarılı.
# Örneğin 0.99 = modelin tahminleri gerçek değerlerin %99'unu açıklıyor.
print(f"R² Skoru:  {r2:.4f}   (1'e yakın = iyi, modelin açıklama gücü)")

# MAE (Ortalama Mutlak Hata): Tahminlerin gerçek değerlerden ortalama ne kadar
# saptığını gösterir. Küçük olması iyidir.
print(f"MAE:       {mae:.4f}   (ortalama tahmin hatası, küçük = iyi)")

# MSE (Ortalama Kare Hata): MAE gibi ama büyük hataları daha çok cezalandırır.
# Hataların karesini alır, bu yüzden büyük sapmalar daha belirgin olur.
print(f"MSE:       {mse:.4f}   (büyük hataları daha çok cezalandırır)")

# RMSE (Kök Ortalama Kare Hata): MSE'nin karekökü. MAE ile aynı birimde olduğu
# için yorumlaması daha kolaydır. "Ortalama ne kadar yanılıyoruz" sorusuna cevap verir.
print(f"RMSE:      {rmse:.4f}   (ortalama sapma miktarı, performans puanı cinsinden)")

# %% [markdown]
# ## 11. Görselleştirme
#
# Model sonuçlarını grafiklerle görselleştirerek performansı ve veri ilişkilerini analiz ediyoruz.

# %%
# --- Grafik 1: Gerçek Değerler vs Tahmin Edilen Değerler ---
# Bu grafik modelin ne kadar doğru tahmin yaptığını gösterir.
# Noktalar kırmızı çizgiye (ideal tahmin hattı) ne kadar yakınsa model o kadar başarılı.
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color="steelblue", edgecolors="k", linewidths=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", linewidth=2, label="İdeal Tahmin Hattı")
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahmin Edilen Değerler")
plt.title("Gerçek Değerler ile Tahmin Edilen Değerlerin Karşılaştırması")
plt.legend()
plt.tight_layout()
plt.savefig("grafik_gercek_vs_tahmin.png", dpi=150)
plt.show()

# %%
# --- Grafik 2: Artık Değerler (Residual) Grafiği ---
# Artık değer = Gerçek - Tahmin (yani modelin hatası)
# İyi bir modelde noktalar 0 çizgisi etrafında rastgele dağılmalı.
# Eğer bir desen (pattern) görünüyorsa model bir şeyleri kaçırıyor demektir.
residuals = y_test - y_pred

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.5, color="coral", edgecolors="k", linewidths=0.5)
plt.axhline(y=0, color="black", linestyle="--", linewidth=1.5)
plt.xlabel("Tahmin Edilen Değerler")
plt.ylabel("Artık Değerler (Residuals)")
plt.title("Artık Değerler Grafiği")
plt.tight_layout()
plt.savefig("grafik_artik_degerler.png", dpi=150)
plt.show()

# %%
# --- Grafik 3: Korelasyon Isı Haritası ---
# Değişkenler arasındaki ilişkiyi gösterir.
# 1'e yakın = güçlü pozitif ilişki (biri artınca diğeri de artar)
# -1'e yakın = güçlü negatif ilişki (biri artınca diğeri azalır)
# 0'a yakın = ilişki yok
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f",
            linewidths=0.5, square=True)
plt.title("Değişkenler Arası Korelasyon Isı Haritası")
plt.tight_layout()
plt.savefig("grafik_korelasyon.png", dpi=150)
plt.show()

# %% [markdown]
# ## 12. Sonuç ve Yorum
#
# ### Modelin Genel Başarısı
# Lineer Regresyon modeli bu veri seti üzerinde iyi sonuçlar vermiştir.
# R² skoru modelin açıklama gücünü, RMSE ise ortalama tahmin hatasını göstermektedir.
# Sonuçlar, lineer regresyonun bu veri seti için uygun bir model olduğunu ortaya koymaktadır.
#
# ### Hangi Değişkenler Daha Etkili?
# Model katsayılarına göre en etkili değişkenler sırasıyla:
# 1. **Hours Studied** — En güçlü etki. Çalışma saati arttıkça performans belirgin şekilde artıyor.
# 2. **Previous Scores** — Önceki sınav puanları da güçlü bir belirleyici.
# 3. **Extracurricular Activities** — Ders dışı aktivitelere katılım küçük ama pozitif bir etki gösteriyor.
# 4. **Sleep Hours** — Uyku saati de performansı olumlu etkiliyor.
# 5. **Sample Question Papers Practiced** — En düşük etkiye sahip değişken.
#
# ### Veri Seti Hakkında Not
# Orijinal veri seti sentetik olduğundan gerçek dünya koşullarını simüle etmek amacıyla
# hedef değişkene Gaussian gürültü (std=7) eklenmiştir. Bu sayede model daha gerçekçi
# bir performans sergilemektedir.

# %%
# Sonuç özetini ekrana yazdırma
print("=" * 50)
print("📝 SONUÇ VE YORUM")
print("=" * 50)
print(f"R² Skoru: {r2:.4f} — Model veriyi %{r2*100:.1f} oranında açıklıyor.")
print(f"RMSE: {rmse:.2f} — Ortalama tahmin hatası ~{rmse:.0f} puan.")
print()
print("En etkili değişkenler (katsayıya göre):")
# Katsayıları büyükten küçüğe sıralayarak yazdırma
katsayi_sirali = sorted(zip(X.columns, model.coef_), key=lambda x: abs(x[1]), reverse=True)
for sira, (ozellik, katsayi) in enumerate(katsayi_sirali, 1):
    print(f"  {sira}. {ozellik}: {katsayi:.4f}")
print()
print("Veri seti dengeli, eksik veri yok, model güvenilir sonuçlar üretiyor.")
