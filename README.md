# Modified K-Nearest Neighbors (MKNN)

MKNN memperkenalkan beberapa perubahan terhadap KNN standar, seperti validasi hasil prediksi, penggunaan bobot untuk memperhitungkan kualitas prediksi, dan fungsi tambahan untuk membandingkan performa MKNN dengan KNN standar.

## Cara Penggunaan

Untuk menggunakan kelas MKNN, ikuti langkah-langkah di bawah ini:

1. **Instalasi**

    Pastikan Anda telah menginstal Python dan paket-paket pendukungnya seperti NumPy dan scikit-learn.

2. **Import**

    Import kelas MKNN ke dalam proyek Python Anda:

    ```python
    from mknn import MKNN
    ```

3. **Inisialisasi Model**

    Buat objek MKNN dengan parameter yang sesuai, seperti jumlah tetangga (n_neighbors) dan metrik jarak (distance):

    ```python
    mknn = MKNN(n_neighbors=5, distance='euclidean')
    ```

4. **Fit Model**

    Fit model menggunakan data pelatihan (X_train) dan label (y_train):

    ```python
    mknn.fit(X_train, y_train)
    ```

5. **Prediksi**

    Lakukan prediksi pada data uji (X_test):

    ```python
    predictions = mknn.predict(X_test)
    ```

6. **Evaluasi**

    Evaluasi performa model dengan menggunakan metrik-metrik evaluasi yang sesuai seperti akurasi:

    ```python
    accuracy = mknn.score(X_test, y_test)
    ```

7. **Perbandingan dengan KNN Standar**

    Untuk membandingkan performa MKNN dengan KNN standar, gunakan fungsi `compare_with_knn`:

    ```python
    comparison = mknn.compare_with_knn(X_test, y_test)
    ```

    Hasilnya akan berupa sebuah dictionary yang berisi akurasi MKNN dan KNN standar.

## Catatan

Pastikan untuk menginstal semua dependensi yang diperlukan sebelum menggunakan kelas MKNN. Juga, pastikan bahwa data yang Anda gunakan telah diproses dengan benar sesuai dengan kebutuhan algoritma.
