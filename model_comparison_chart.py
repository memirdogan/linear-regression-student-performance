# %% [markdown]
# # Model Karşılaştırma Grafiği — Poster İçin

# %%
import matplotlib.pyplot as plt
import numpy as np

# Model isimleri ve metrikler
modeller = ["Lineer Regresyon", "Random Forest"]
r2_degerleri = [0.8705, 0.8430]  # R² değerleri
rmse_degerleri = [7.31, 8.05]    # RMSE değerleri

# Renkler
renkler = ["#3498db", "#27ae60"]  # Mavi ve yeşil

# Grafik 1: R² Karşılaştırması
fig, ax = plt.subplots(figsize=(8, 6))

x = np.arange(len(modeller))
genislik = 0.6

barlar = ax.bar(x, r2_degerleri, genislik, color=renkler, edgecolor="black", linewidth=1.5)

# Değerleri barların üstüne yaz
for bar, deger in zip(barlar, r2_degerleri):
    height = bar.get_height()
    ax.annotate(f'{deger:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=14, fontweight="bold")

ax.set_ylabel("R² Skoru", fontsize=12)
ax.set_title("Model Karşılaştırması — R² Skoru", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(modeller, fontsize=12)
ax.set_ylim(0, 1.0)
ax.grid(axis="y", alpha=0.3)
ax.axhline(y=0.8, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Kabul edilebilir eşik (0.8)")
ax.legend()

plt.tight_layout()
plt.savefig("poster_model_r2_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# Grafik 2: RMSE Karşılaştırması
fig, ax = plt.subplots(figsize=(8, 6))

barlar2 = ax.bar(x, rmse_degerleri, genislik, color=renkler, edgecolor="black", linewidth=1.5)

for bar, deger in zip(barlar2, rmse_degerleri):
    height = bar.get_height()
    ax.annotate(f"{deger:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=14, fontweight="bold")

ax.set_ylabel("RMSE", fontsize=12)
ax.set_title("Model Karşılaştırması — RMSE (Düşük = İyi)", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(modeller, fontsize=12)
ax.set_ylim(0, 10)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("poster_model_rmse_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

print("Grafikler kaydedildi:")
print("  - poster_model_r2_comparison.png")
print("  - poster_model_rmse_comparison.png")