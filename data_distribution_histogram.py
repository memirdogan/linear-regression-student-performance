# %% [markdown]
# # Veri Dağılımı Histogramları — Poster İçin
#
# Bu grafik, Student Performance veri setindeki tüm değişkenlerin dağılımını
# gösterir. Lineer Regresyon ve Random Forest modelleri için ortak kullanılabilir.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Veri yükleme ve gürültü ekleme (her iki notebook ile aynı)
df = pd.read_csv("Student_Performance.csv")
df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})

np.random.seed(42)
df["Performance Index"] = (df["Performance Index"] + np.random.normal(0, 7, size=len(df))).clip(0, 100)

# Grafik oluşturma
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Veri Seti Değişken Dağılımları (Data Distribution)", fontsize=16, fontweight="bold")

# Her değişken için histogram
columns = ["Hours Studied", "Previous Scores", "Extracurricular Activities", 
           "Sleep Hours", "Sample Question Papers Practiced", "Performance Index"]

for idx, col in enumerate(columns):
    ax = axes[idx // 3, idx % 3]
    ax.hist(df[col], bins=20, color="steelblue", edgecolor="black", alpha=0.7)
    ax.set_title(col, fontsize=12, fontweight="bold")
    ax.set_xlabel("Değer")
    ax.set_ylabel("Frekans")
    ax.grid(axis="y", alpha=0.3)

# 6. grafik alanını boşalt (sadece 5 değişken var)
axes[1, 2].axis("off")

plt.tight_layout()
plt.savefig("poster_data_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

print("Grafik 'poster_data_distribution.png' olarak kaydedildi.")