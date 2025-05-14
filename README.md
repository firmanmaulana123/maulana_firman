# 🌸 Visualisasi dan Analisis Dataset Iris - Tugas AI Kelompok A

Repositori ini berisi notebook Jupyter yang dibuat untuk tugas mata kuliah **Kecerdasan Buatan**, dengan studi kasus pada **dataset Iris**. Notebook ini mencakup visualisasi data dan penerapan teknik **Principal Component Analysis (PCA)** untuk reduksi dimensi.

## 🧾 Isi Notebook

Notebook ini berjudul **Kelompok A_Tugas AI.ipynb** dan mencakup:

1. **Pemuatan dataset Iris** menggunakan `sklearn.datasets`.
2. **Visualisasi pairplot** menggunakan Seaborn untuk menampilkan hubungan antar fitur dengan pewarnaan berdasarkan kelas target.
3. **Reduksi dimensi menggunakan PCA (Principal Component Analysis)** menjadi 3 komponen utama.
4. **Visualisasi 3D** hasil PCA menggunakan Matplotlib.

## 🧪 Tools dan Library yang Digunakan

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## 📊 Tujuan

Notebook ini bertujuan untuk:
- Menjelaskan bagaimana visualisasi data membantu memahami distribusi dan klasifikasi.
- Menerapkan PCA sebagai teknik unsupervised learning untuk mereduksi dimensi dataset.
- Menampilkan data dalam bentuk 3D untuk pemahaman yang lebih intuitif.

## 🚀 Cara Menjalankan

1. Clone repositori:
   ```bash
   git clone https://github.com/username/nama-repo.git
   cd nama-repo
   ```

2. (Opsional) Buat dan aktifkan virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # Linux/Mac
   env\Scripts\activate     # Windows
   ```

3. Instal dependensi:
   ```bash
   pip install -r requirements.txt
   ```

4. Jalankan notebook:
   ```bash
   jupyter notebook Kelompok\ A_Tugas\ AI.ipynb
   ```

## 📦 `requirements.txt`

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

## 👨‍👩‍👧‍👦 Anggota Kelompok

- Firman Maulana Putra – 24293014  
- Moh. Fikry Al-Farisy – 24293041

## 📚 Lisensi

Proyek ini dibuat untuk tujuan pembelajaran dan akademik.

# 1. Import library yang dibutuhkan
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 2. Load dataset Iris
iris = load_iris()
X = iris.data  # Fitur: panjang dan lebar sepal, panjang dan lebar petal
y = iris.target  # Target: jenis bunga

# 3. Buat DataFrame untuk visualisasi dan eksplorasi
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
df['target_name'] = df['target'].apply(lambda i: iris.target_names[i])

# 4. Visualisasi data
sns.pairplot(df, hue='target_name')
plt.suptitle('Visualisasi Fitur Iris', y=1.02)
plt.show()

# 5. Split data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Normalisasi fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Inisialisasi dan latih model (contoh: Logistic Regression)
model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)

# 8. Prediksi dan evaluasi
y_pred = model.predict(X_test_scaled)

# 9. Tampilkan hasil evaluasi
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# 10. Visualisasi Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
