import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Membuat DataFrame dari data yang diberikan
data = {
    'Advertising': [12.0, 20.5, 21.0, 15.5, 15.3, 23.5, 24.5, 21.3, 23.5, 28.0, 24.0, 15.5, 17.3, 25.3, 25.0, 36.5, 36.5, 29.6, 30.5, 28.0, 26.0, 21.5, 19.7, 19.0, 16.0, 20.7, 26.5, 30.6, 32.3, 29.5, 28.3, 31.3, 32.3, 26.4, 23.4, 16.4],
    'Sales': [15.0, 16.0, 18.0, 27.0, 21.0, 49.0, 21.0, 22.0, 28.0, 36.0, 40.0, 3.0, 21.0, 29.0, 62.0, 65.0, 46.0, 44.0, 33.0, 62.0, 22.0, 12.0, 24.0, 3.0, 5.0, 14.0, 36.0, 40.0, 49.0, 7.0, 52.0, 65.0, 17.0, 5.0, 17.0, 1.0]
}
df = pd.DataFrame(data)

# Membagi data menjadi fitur (X) dan target (y)
X = df[['Sales']]
y = df['Advertising']

# Membuat dan melatih model regresi linear
model = LinearRegression()
model.fit(X, y)

# Memprediksi biaya iklan untuk target penjualan 50, 100, 150
sales_targets = pd.DataFrame({'Sales': [50, 100, 150]})
predictions = model.predict(sales_targets)

# Menghitung RMSE dan R2 score
y_pred = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

# Menampilkan hasil prediksi dan evaluasi
print(f"Prediksi biaya iklan untuk 50 sales: {predictions[0]:.2f} million $")
print(f"Prediksi biaya iklan untuk 100 sales: {predictions[1]:.2f} million $")
print(f"Prediksi biaya iklan untuk 150 sales: {predictions[2]:.2f} million $")
print(f"RMSE: {rmse:.2f}")
print(f"R2 score: {r2:.2f}")

# Visualisasi hasil regresi
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red', linewidth=2)
plt.xlabel('Sales (million $)')
plt.ylabel('Advertising (million $)')
plt.title('Sales vs Advertising')
plt.show()
