# %% [markdown]
# # Artık (Residual) Değerler Grafiği — Poster İçin

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Veri yükleme ve işleme
df = pd.read_csv("Student_Performance.csv")
df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})

np.random.seed(42)
df["Performance Index"] = (df["Performance Index"] + np.random.normal(0, 7, size=len(df))).clip(0, 100)

X = df[["Hours Studied", "Previous Scores", "Extracurricular Activities", 
        "Sleep Hours", "Sample Question Papers Practiced"]]
y = df["Performance Index"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lineer Regresyon
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_residuals = y_test - lr_pred

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_residuals = y_test - rf_pred

# Grafik oluşturma
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Şekil 5: Artık (Residual) Değerler Grafiği", fontsize=16, fontweight="bold", y=1.02)

# Lineer Regresyon Artıkları
axes[0].scatter(lr_pred, lr_residuals, alpha=0.5, color="#3498db", edgecolors="k", linewidths=0.5)
axes[0].axhline(y=0, color="red", linestyle="--", linewidth=2)
axes[0].set_xlabel("Tahmin Edilen Değerler", fontsize=11)
axes[0].set_ylabel("Artık Değerler (Gerçek - Tahmin)", fontsize=11)
axes[0].set_title("Lineer Regresyon", fontsize=13, fontweight="bold")
axes[0].grid(alpha=0.3)
axes[0].set_xlim([10, 95])
axes[0].set_ylim([-25, 25])

# Random Forest Artıkları
axes[1].scatter(rf_pred, rf_residuals, alpha=0.5, color="#27ae60", edgecolors="k", linewidths=0.5)
axes[1].axhline(y=0, color="red", linestyle="--", linewidth=2)
axes[1].set_xlabel("Tahmin Edilen Değerler", fontsize=11)
axes[1].set_ylabel("Artık Değerler (Gerçek - Tahmin)", fontsize=11)
axes[1].set_title("Random Forest", fontsize=13, fontweight="bold")
axes[1].grid(alpha=0.3)
axes[1].set_xlim([10, 95])
axes[1].set_ylim([-25, 25])

plt.tight_layout()
plt.savefig("poster_residual_plot.png", dpi=300, bbox_inches="tight")
plt.show()

print("Grafik 'poster_residual_plot.png' olarak kaydedildi.")
print("\nAlt Yazı: Artık değerlerin sıfır etrafında rastgele dağılması modelin uygun olduğunu göstermektedir.")