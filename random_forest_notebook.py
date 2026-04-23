# %% [markdown]
# # Random Forest ile Öğrenci Performans Tahmini
#
# Bu notebook, **Student_Performance.csv** veri seti üzerinde öğrenci performansını
# tahmin eden bir Random Forest Regresyon modeli geliştirmektedir. Veri yükleme, keşif,
# ön işleme, model eğitimi, değerlendirme ve görselleştirme adımlarını içerir.
#
# ## Random Forest Nedir?
#
# Random Forest, birden fazla karar ağacını (decision tree) bir araya getiren bir
# **topluluk öğrenme (ensemble learning)** yöntemidir. Her ağaç verinin rastgele bir
# alt kümesiyle eğitilir ve son tahmin tüm ağaçların tahminlerinin ortalaması alınarak
# yapılır. Bu sayede tek bir karar ağacına göre daha kararlı ve doğru sonuçlar üretir.
#
# **Lineer Regresyondan farkı:** Lineer regresyon sadece doğrusal ilişkileri yakalar,
# Random Forest ise doğrusal olmayan (non-linear) ilişkileri de öğrenebilir.

# %% 
# Gerekli kütüphanelerin içe aktarılması
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# %% [markdown]
# ## 1. Veri Yükleme

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
# Lineer regresyon notebook'undaki ile aynı gürültü ekleniyor.
# Aynı random seed kullanıldığı için her iki modelde de aynı veri seti üzerinde çalışılır.

# %%
# Hedef değişkene rastgele gürültü ekleme
np.random.seed(42)
gurultu = np.random.normal(0, 7, size=len(df))
df["Performance Index"] = df["Performance Index"] + gurultu
df["Performance Index"] = df["Performance Index"].clip(0, 100)

print("=" * 50)
print("🔊 GÜRÜLTÜ EKLEME SONUCU")
print("=" * 50)
print(f"Eklenen gürültü: Gaussian(ortalama=0, std=7)")
print(f"Yeni Performance Index aralığı: {df['Performance Index'].min():.1f} - {df['Performance Index'].max():.1f}")

# %% [markdown]
# ## 3. Veri Keşfi

# %%
print("=" * 50)
print("📋 İLK 5 SATIR (head)")
print("=" * 50)
print(df.head())

# %%
print("=" * 50)
print("ℹ️  VERİ SETİ BİLGİSİ (info)")
print("=" * 50)
df.info()

# %%
print("=" * 50)
print("📊 İSTATİSTİKSEL ÖZET (describe)")
print("=" * 50)
print(df.describe())

# %% [markdown]
# ## 4. Eksik Veri Kontrolü

# %%
eksik_veriler = df.isnull().sum()
print("=" * 50)
print("🔍 EKSİK VERİ KONTROLÜ")
print("=" * 50)
print(eksik_veriler)
print(f"\nToplam eksik değer: {eksik_veriler.sum()}")

# %% [markdown]
# ## 5. Veri Ön İşleme

# %%
# Yes/No → 1/0 dönüşümü
df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})

print("=" * 50)
print("🔄 KATEGORİK DÖNÜŞÜM SONUCU")
print("=" * 50)
print(f"Benzersiz değerler: {df['Extracurricular Activities'].unique()}")
print(f"Veri tipi: {df['Extracurricular Activities'].dtype}")

# %% [markdown]
# ## 6. Özellik ve Hedef Değişken Tanımlama

# %%
X = df[["Hours Studied", "Previous Scores", "Extracurricular Activities",
        "Sleep Hours", "Sample Question Papers Practiced"]]
y = df["Performance Index"]

print("=" * 50)
print("📐 ÖZELLİK VE HEDEF DEĞİŞKEN BOYUTLARI")
print("=" * 50)
print(f"X (girdi verileri): {X.shape[0]} satır, {X.shape[1]} sütun")
print(f"y (tahmin edilecek değer): {y.shape[0]} satır")

# %% [markdown]
# ## 7. Veri Bölme

# %%
# Aynı random_state kullanarak lineer regresyon ile aynı bölmeyi elde ediyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("=" * 50)
print("✂️  EĞİTİM / TEST BÖLME SONUCU")
print("=" * 50)
print(f"Eğitim seti: {X_train.shape[0]} satır")
print(f"Test seti:   {X_test.shape[0]} satır")

# %% [markdown]
# ## 8. Model Eğitimi — Random Forest
#
# Random Forest modeli birden fazla karar ağacı oluşturur ve tahminlerinin
# ortalamasını alır. `n_estimators` parametresi kaç ağaç kullanılacağını belirler.
# Varsayılan değer 100 ağaçtır.

# %%
# Random Forest modeli oluşturma ve eğitme
# n_estimators=100: 100 karar ağacı kullanılacak
# random_state=42: tekrarlanabilirlik için sabit tohum
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Özellik önem dereceleri (feature importances)
# Her değişkenin tahmine ne kadar katkı sağladığını gösterir (toplam = 1.0)
# Lineer regresyondaki katsayılardan farklı olarak, burada önem yüzde olarak ifade edilir
print("=" * 50)
print("🌲 RANDOM FOREST — ÖZELLİK ÖNEM DERECELERİ")
print("=" * 50)
print("(Her değişkenin tahmine katkısı, toplam = 1.0)")
print()
for ozellik, onem in sorted(zip(X.columns, model.feature_importances_),
                              key=lambda x: x[1], reverse=True):
    print(f"  {ozellik}: {onem:.4f} (%{onem*100:.1f})")

# %% [markdown]
# ## 9. Tahmin

# %%
# Test verisi üzerinde tahmin üretme
y_pred = model.predict(X_test)

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

# %%
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("=" * 50)
print("📈 PERFORMANS METRİKLERİ — RANDOM FOREST")
print("=" * 50)
print(f"R² Skoru:  {r2:.4f}")
print(f"MAE:       {mae:.4f}")
print(f"MSE:       {mse:.4f}")
print(f"RMSE:      {rmse:.4f}")

# %% [markdown]
# ## 11. Görselleştirme

# %%
# Grafik 1: Gerçek vs Tahmin
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color="forestgreen", edgecolors="k", linewidths=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", linewidth=2, label="İdeal Tahmin Hattı")
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahmin Edilen Değerler")
plt.title("Random Forest — Gerçek vs Tahmin Karşılaştırması")
plt.legend()
plt.tight_layout()
plt.savefig("rf_grafik_gercek_vs_tahmin.png", dpi=150)
plt.show()

# %%
# Grafik 2: Artık Değerler
residuals = y_test - y_pred

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.5, color="darkorange", edgecolors="k", linewidths=0.5)
plt.axhline(y=0, color="black", linestyle="--", linewidth=1.5)
plt.xlabel("Tahmin Edilen Değerler")
plt.ylabel("Artık Değerler (Residuals)")
plt.title("Random Forest — Artık Değerler Grafiği")
plt.tight_layout()
plt.savefig("rf_grafik_artik_degerler.png", dpi=150)
plt.show()

# %%
# Grafik 3: Korelasyon Isı Haritası
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f",
            linewidths=0.5, square=True)
plt.title("Değişkenler Arası Korelasyon Isı Haritası")
plt.tight_layout()
plt.savefig("rf_grafik_korelasyon.png", dpi=150)
plt.show()

# %%
# Grafik 4: Özellik Önem Dereceleri (Random Forest'a özel)
plt.figure(figsize=(8, 5))
onem_sirali = sorted(zip(X.columns, model.feature_importances_), key=lambda x: x[1], reverse=True)
ozellikler = [x[0] for x in onem_sirali]
degerler = [x[1] for x in onem_sirali]
plt.barh(ozellikler[::-1], degerler[::-1], color="forestgreen", edgecolor="k")
plt.xlabel("Önem Derecesi")
plt.title("Random Forest — Özellik Önem Dereceleri")
plt.tight_layout()
plt.savefig("rf_grafik_ozellik_onem.png", dpi=150)
plt.show()

# %% [markdown]
# ## 12. Sonuç ve Yorum
#
# ### Modelin Genel Başarısı
# Random Forest modeli, birden fazla karar ağacının gücünü birleştirerek tahmin yapmaktadır.
# Sonuçlar lineer regresyon ile karşılaştırılarak modellerin güçlü ve zayıf yönleri
# değerlendirilebilir.
#
# ### Özellik Önem Dereceleri
# Random Forest, her değişkenin tahmine ne kadar katkı sağladığını doğrudan ölçebilir.
# Bu, lineer regresyondaki katsayılardan farklı bir bakış açısı sunar.
#
# ### Veri Seti Hakkında Not
# Lineer regresyon notebook'undaki ile aynı gürültü (std=7) ve aynı veri bölmesi
# (random_state=42) kullanılmıştır. Bu sayede iki model adil bir şekilde karşılaştırılabilir.

# %%
print("=" * 50)
print("📝 SONUÇ VE YORUM — RANDOM FOREST")
print("=" * 50)
print(f"R² Skoru: {r2:.4f} — Model veriyi %{r2*100:.1f} oranında açıklıyor.")
print(f"RMSE: {rmse:.2f} — Ortalama tahmin hatası ~{rmse:.0f} puan.")
print()
print("Özellik önem dereceleri (büyükten küçüğe):")
for sira, (ozellik, onem) in enumerate(onem_sirali, 1):
    print(f"  {sira}. {ozellik}: %{onem*100:.1f}")
