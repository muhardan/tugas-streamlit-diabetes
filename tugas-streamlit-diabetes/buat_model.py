import pandas as pd
from sklearn.svm import SVC
import pickle
import os

print("Memulai proses pembuatan model lokal...")

# 1. Pastikan file diabetes.csv ada
if not os.path.exists('diabetes.csv'):
    print("❌ ERROR: File 'diabetes.csv' tidak ditemukan di folder ini!")
    print("Pastikan file dataset sudah satu folder dengan file ini.")
else:
    # 2. Baca dataset
    print("Membaca data diabetes.csv...")
    df = pd.read_csv('diabetes.csv')

    # 3. Pisahkan fitur dan target
    X = df.drop(columns='Outcome', axis=1)
    y = df['Outcome']

    # 4. Latih model
    print("Melatih model Machine Learning... (Tunggu sebentar)")
    model = SVC(kernel='linear')
    model.fit(X, y)

    # 5. Simpan model
    filename = 'diabetes_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    print(f"✅ BERHASIL! File '{filename}' versi lokal sudah dibuat dan siap digunakan.")