# Predicting Purchases with Google Analytics Session Data

**Pemilik Repositori:** Roofiif Alria Dzakwan
**NRP:** 5025221012

## Deskripsi Proyek

Repositori ini berisi kode dan sumber daya untuk proyek prediksi pembelian berdasarkan data sesi Google Analytics. Tujuan dari proyek ini adalah membangun model *machine learning* yang dapat memprediksi apakah seorang pengguna akan melakukan pembelian pada kunjungan berikutnya berdasarkan informasi dari sesi saat ini.

Proyek ini berkaitan dengan tugas:

> 🛍️ Predicting Purchases with Google Analytics Session Data
> 📄 Overview
> Dataset ini mensimulasikan perilaku pengguna di situs web *e-commerce* menggunakan data tingkat sesi Google Analytics. Setiap baris mewakili sesi unik, dengan berbagai atribut perilaku, sumber lalu lintas, dan terkait perangkat. Tujuannya adalah untuk memprediksi apakah pengguna akan melakukan pembelian pada kunjungan kembali, berdasarkan informasi dari sesi saat ini.
>
> Jenis masalah ini umum dalam tugas pemasaran digital, analisis produk, dan optimasi konversi di dunia nyata.
>
> 🎯 Objective
> Tugas Anda adalah membangun model yang memprediksi hasil biner:
> Akankah pengguna melakukan pembelian jika mereka kembali pada sesi berikutnya?
> Ini dapat membantu bisnis memahami niat pengunjung, mempersonalisasi pengalaman pengguna, dan menargetkan ulang lalu lintas berkualitas tinggi.
>
> 📌 Target Column
> `will_buy_on_return_visit`: Label klasifikasi biner:
> `1`: Pengguna kembali kemudian dan melakukan pembelian.
> `0`: Pengguna tidak kembali atau tidak melakukan konversi.
>
> 🧾 Features
>
> | Feature Name              | Description                                                              |
> |---------------------------|--------------------------------------------------------------------------|
> | `unique_session_id`       | Pengenal unik untuk setiap sesi (berdasarkan `fullVisitorId` dan `visitId`). |
> | `bounces`                 | Menunjukkan apakah sesi berakhir dengan hanya satu tampilan halaman.       |
> | `time_on_site`            | Total waktu yang dihabiskan di situs selama sesi (dalam detik).          |
> | `pageviews`               | Jumlah halaman yang dilihat selama sesi.                               |
> | `hits`                    | Total jumlah interaksi pengguna selama sesi.                           |
> | `session_quality_dim`     | Skor kualitas kepemilikan (1–100) yang memperkirakan niat pembelian.      |
> | `latest_ecommerce_progress`| Tahap terjauh yang dicapai dalam *e-commerce funnel*.                   |
> | `avg_time_per_page`       | Rata-rata waktu per halaman (dihitung sebagai `time_on_site` / `pageviews`). |
> | `source`                  | Sumber lalu lintas (misalnya, google, bing, direct).                   |
> | `medium`                  | Medium lalu lintas (misalnya, organic, cpc, referral).                  |
> | `channelGrouping`         | Kategori saluran tingkat tinggi (misalnya, Referral, Organic Search, Direct). |
> | `deviceCategory`          | Kategori perangkat yang digunakan (desktop, tablet, mobile).           |
> | `operatingSystem`         | Kategori sistem operasi perangkat yang digunakan.                       |
> | `browser`                 | Kategori *browser* perangkat yang digunakan.                            |
> | `country`                 | Negara dari mana pengguna mengakses situs.                             |
> | `city`                    | Kota dari mana pengguna mengakses situs.                               |
>
> 📂 Files
>
> | File               | Description                                   |
> |--------------------|-----------------------------------------------|
> | `train.csv`        | Data pelatihan yang tersedia bagi peserta.    |
> | `test.csv`         | Data *private leaderboard* yang tersembunyi. |
> | `sample_submission.csv`| Contoh format pengiriman.                   |
>
> 🧪 Evaluation Metric
>
> Submissions dievaluasi berdasarkan skor akurasi antara probabilitas prediksi dan target yang diamati.
>
> 📤 Submission File
>
> Untuk setiap ID dalam set pengujian, Anda harus memprediksi probabilitas untuk variabel TARGET. File harus berisi *header* dan memiliki format berikut:
>
> ```csv
> ID,TARGET
> 205,0.0123
> 06,0.9876
> etc.
> ```
>
> **Catatan:** Berdasarkan interaksi sebelumnya, pengiriman aktual ke Kaggle memerlukan prediksi kelas biner (0 atau 1), bukan probabilitas. Kode dalam repositori ini akan menghasilkan probabilitas, dan Anda mungkin perlu menyesuaikannya untuk menghasilkan kelas biner dengan *threshold* 0.5 sebelum mengirimkannya ke Kaggle.

## Struktur Repositori

e-commerce-purchase-prediction/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── notebooks/
│   ├── EDA.ipynb           # Notebook untuk Exploratory Data Analysis
│   ├── model_training.ipynb # Notebook untuk pelatihan model
│   └── ...
├── src/
│   ├── preprocessing.py  # Skrip untuk fungsi preprocessing
│   ├── model.py        # Skrip untuk definisi dan pelatihan model
│   └── ...
├── submissions/
│   ├── submission.csv      # Contoh file pengiriman
│   ├── submission_bgm.csv
│   ├── submission_gbm.csv
│   └── submission_catboost_optuna_gpu.csv
├── README.md
└── requirements.txt      # Daftar dependensi Python


## Cara Menggunakan

1.  **Kloning Repositori:**
    ```bash
    git clone [https://github.com/roofiif-alria-dzakwan/e-commerce-purchase-prediction.git](https://github.com/roofiif-alria-dzakwan/e-commerce-purchase-prediction.git)
    cd e-commerce-purchase-prediction
    ```

2.  **Instal Dependensi:**
    ```bash
    pip install -r requirements.txt
    ```
    (Anda perlu membuat file `requirements.txt` yang berisi daftar *library* Python yang digunakan, seperti `pandas`, `scikit-learn`, `catboost`, `lightgbm`, `xgboost`, `optuna`, `tqdm`, `torch`). Contoh `requirements.txt`:
    ```
    pandas
    scikit-learn
    catboost
    lightgbm
    xgboost
    optuna
    tqdm
    torch
    ```

3.  **Jelajahi Notebook:**
    * `EDA.ipynb`: Berisi analisis eksplorasi data untuk memahami dataset.
    * `model_training.ipynb`: Berisi kode untuk melatih berbagai model (*baseline*, CatBoost, LightGBM, XGBoost, *ensemble*), termasuk *hyperparameter tuning*.

4.  **Jalankan Skrip:**
    * Skrip di direktori `src/` dapat dijalankan untuk melakukan *preprocessing*, pelatihan model, dan menghasilkan file pengiriman secara modular.

## Hasil dan Pengiriman

File pengiriman (`submission.csv` atau nama file lain yang dihasilkan) akan disimpan di direktori `submissions/`. File ini dapat diunggah ke platform kompetisi untuk evaluasi.

## Catatan Tambahan

* Kode dalam repositori ini mencakup eksperimen dengan beberapa model, termasuk CatBoost yang teroptimasi menggunakan Optuna dan dukungan GPU (jika tersedia).
* Eksplorasi fitur lebih lanjut (*feature engineering*) dan *tuning hyperparameter* yang lebih mendalam dapat dilakukan untuk meningkatkan performa model.
* Pendekatan *ensemble* (seperti *stacking*) juga dieksplorasi untuk potensi peningkatan akurasi.
