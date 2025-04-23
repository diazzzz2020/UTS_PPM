
1.Persiapan Data
- Import data CSV.
- Eksplorasi data awal: Melihat beberapa baris awal data, tipe data, jumlah missing values.
- Membersihkan data:
  - Menangani missing values.
  - Menghapus atau mengisi nilai yang tidak valid.
- Encoding data kategorikal (jika ada) agar dapat digunakan oleh algoritma (misalnya dengan Label Encoding).

2. Analisis dan Visualisasi Data
- Melihat distribusi kelas target.
- Mengecek korelasi antar fitur (jika numeric).
- Visualisasi data (heatmap).

3. Preprocessing Data
- Split data menjadi fitur (`X`) dan target (`y`).
- Split data latih dan data uji, misalnya 80:20 atau 70:30, dengan `train_test_split` dari `sklearn`.
- Standarisasi atau normalisasi data numerik

4. Pelatihan Model
- Menggunakan `GaussianNB`.
  Contoh:
  from sklearn.naive_bayes import GaussianNB
  model = GaussianNB()
  model.fit(X_train, y_train)


5. Prediksi
- Menggunakan model yang sudah dilatih untuk memprediksi data uji:
  y_pred = model.predict(X_test)
  

6. Evaluasi Model
- Menggunakan metrik evaluasi:
  - akurasi: `accuracy_score`
  - Precision, Recall, F1-score: `classification_report`
  - Confusion Matrix: `confusion_matrix`
  - Visualisasi: heatmap confusion matrix

  Contoh:
  from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
  print("Akurasi:", accuracy_score(y_test, y_pred))
  print(classification_report(y_test, y_pred))
  print(confusion_matrix(y_test, y_pred))
