# %% [markdown]
# # Predictive Analytics Employees' Burnout

# %% [markdown]
# # 1. Data Loading

# %% [markdown]
# ## 1.1. Kaggle Credentials

# %% [markdown]
# Kaggle Username dan Kaggle Key diperlukan untuk mengakses dataset pada Kaggle. Kedua variabel tersebut kemudian disimpan dalam environment variable dengan bantuan library `os`.

# %%
import os
os.environ['KAGGLE_USERNAME'] = 'fahmial'
os.environ['KAGGLE_KEY'] = 'b34007481a89d1b149bccd090f285846'

# %% [markdown]
# ## 1.2. Download the Dataset

# %% [markdown]
# Dataset yang digunakan adalah [Are Your Employees Burning Out?](https://www.kaggle.com/datasets/blurredmachine/are-your-employees-burning-out) dengan `train.csv` sebagai dataset training dan validation.

# %%
# Download train.csv ke local directory
!kaggle datasets download -d blurredmachine/are-your-employees-burning-out -f train.csv -p .

# %%
# Extract file
!unzip -qo train.csv.zip && rm train.csv.zip

# %% [markdown]
# # 2. Data Understanding

# %% [markdown]
# Import library yang diperlukan.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

# %% [markdown]
# ## 2.1. Menampilkan Data

# %%
train_df = pd.read_csv('train.csv')
train_df.head()

# %% [markdown]
# Menghapus kolom `Employee ID` karena tidak diperlukan.

# %%
train_df.drop(['Employee ID'], axis=1, inplace=True)

# %% [markdown]
# ## 2.2. Exploratory Data Analysis

# %% [markdown]
# Di sini akan dilakukan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data.

# %% [markdown]
# ### 2.2.1. Deskripsi Variabel

# %% [markdown]
# Pengecekan informasi variabel dari dataset berupa jumlah kolom, nama kolom, jumlah data per kolom dan tipe datanya.

# %%
train_df.info()

# %% [markdown]
# File ini terdiri dari 9 kolom sebagai berikut:
# 
# - Employee ID (telah dihapus): ID unik yang dialokasikan untuk setiap karyawan (contoh: fffe390032003000)
# - Date of Joining: Tanggal-waktu ketika karyawan telah bergabung dengan organisasi (contoh: 2008-12-30)
# - Gender: Jenis kelamin karyawan (Pria/Wanita)
# - Company Type: Jenis perusahaan tempat karyawan bekerja (Layanan/Produk)
# - WFH Setup Available: Apakah fasilitas bekerja dari rumah tersedia untuk karyawan (Ya/Tidak)
# - Designation: Penunjukan karyawan yang bekerja di organisasi.
#   - Di kisaran [0.0, 5.0] lebih besar adalah penunjukan yang lebih tinggi.
# - Resource Allocation: Jumlah sumber daya yang dialokasikan kepada karyawan untuk bekerja, yaitu. jumlah jam kerja.
#   - Dalam kisaran [1.0, 10.0] (lebih tinggi berarti lebih banyak sumber daya)
# - Mental Fatigue Score: Tingkat kelelahan mental yang dihadapi karyawan.
#   - Dalam rentang [0.0, 10.0] dimana 0.0 berarti tidak ada kelelahan dan 10.0 berarti kelelahan total.
# - Burn Rate: Nilai yang perlu kita prediksi untuk setiap karyawan yang memberitahukan tingkat Burn out saat bekerja.
#   - Pada rentang [0.0, 1.0] dimana semakin tinggi nilainya maka semakin banyak terjadi burn out.

# %% [markdown]
# ### 2.2.2. Deskripsi Statistik

# %% [markdown]
# Melakukan pengecekan deskripsi statistik data dengan fitur describe().

# %%
train_df.describe()

# %% [markdown]
# Berdasarkan output diatas, didapatkan deskripsi statistik yaitu:
# 1. count: Jumlah sampel data
# 2. mean: Nilai rata-rata
# 3. std: Standar deviasi
# 4. min: Nilai minimum
# 5. 25%: Kuartil bawah/Q1
# 6. 50%: Kuartil tengah/Q2/median
# 7. 75%: Kuartil atas/Q3
# 8. max: Nilai maksimum

# %% [markdown]
# ### 2.2.3. Menangani Missing Value

# %% [markdown]
# Menemukan nilai yang hilang pada dataset.

# %%
train_df.isnull().sum()

# %% [markdown]
# Menghapus baris yang memiliki missing value.

# %%
train_df.dropna(inplace=True)

# Detail setelah dilakukan penghapusan data yang kosong
train_df.describe()

# %% [markdown]
# ### 2.2.4. Menangani Outliers

# %% [markdown]
# Menemukan outlier pada dataset.

# %%
fig, ax = plt.subplots(2, 2, figsize=(12, 6))

sns.boxplot(ax=ax[0, 0], x=train_df['Designation'])
sns.boxplot(ax=ax[0, 1], x=train_df['Resource Allocation'])
sns.boxplot(ax=ax[1, 0], x=train_df['Mental Fatigue Score'])
sns.boxplot(ax=ax[1, 1], x=train_df['Burn Rate'])


# %% [markdown]
# Berdasarkan boxplot diatas, didapatkan outliers pada variabel `Mental Fatigue Score` dan `Burn Rate`.

# %% [markdown]
# Buat batas bawah dengan rumus `Q1 - 1.5 * IQR` dan batas atas dengan rumus `Q3 + 1.5 * IQR` dengan `IQR = Q3 - Q1`.

# %%
Q1 = train_df.quantile(0.25, numeric_only=True)
Q3 = train_df.quantile(0.75, numeric_only=True)
IQR = Q3 - Q1

left1, right1 = train_df.align(Q1 - 1.5 * IQR, axis=1, copy=False)
left2, right2 = train_df.align(Q3 + 1.5 * IQR, axis=1, copy=False)
left, right = (left1 < right1).align((left2 > right2), axis=1, copy=False)
train_df = train_df[~(left | right).any(axis=1)]

# Detail setelah dilakukan penghapusan data outlier
train_df.describe()



# %% [markdown]
# Hasil setelah dihilangkan outliersnya dan mengatur batas bawah dan batas atasnya.

# %%
fig, ax = plt.subplots(2, 2, figsize=(12, 6))

sns.boxplot(ax=ax[0, 0], x=train_df['Designation'])
sns.boxplot(ax=ax[0, 1], x=train_df['Resource Allocation'])
sns.boxplot(ax=ax[1, 0], x=train_df['Mental Fatigue Score'])
sns.boxplot(ax=ax[1, 1], x=train_df['Burn Rate'])


# %% [markdown]
# ### 2.2.5. Univariate Analysis

# %% [markdown]
# ##### Feature Engineering

# %% [markdown]
# Sebelum membagi fitur menjadi fitur numerik dan fitur kategorik, dilakukan feature engineering terlebih dahulu. Ini dilakukan untuk mengubah fitur `Date of Joining` menjadi fitur `Days Employed` dan menjadikannya sebagai fitur numerik.

# %%
# Cast "Date of Joining" ke datetime
train_df['Date of Joining'] = pd.to_datetime(train_df['Date of Joining'])

# %%
# Dapatkan "Date of Joining" terbaru
max_date = train_df['Date of Joining'].max()

# Buat variabel last_date yang merupakan max_date ditambah 1 hari
last_date = max_date + pd.DateOffset(days=1)

# %% [markdown]
# Kita dapat mengasumsikan bahwa semakin banyak hari yang dimiliki seseorang, maka semakin berpengalaman dia.

# %%
# Buat kolom "Days Employed"
days_employed = last_date - train_df.loc[:, 'Date of Joining']
days_employed = days_employed.dt.days

# Replace 'Date of Joining' dengan 'Days Employed'

# Buat list nama dengan 'Days Employed' di-insert pada posisi yang tepat
column_names = train_df.columns.tolist()
column_names.insert(column_names.index('Date of Joining'), 'Days Employed')
train_df['Days Employed'] = days_employed

# Reorder kolom di DataFrame
train_df = train_df.reindex(columns=column_names)

# Drop kolom 'Date of Joining'
train_df.drop(['Date of Joining'], axis=1, inplace=True)

# %%
# Menampilkan data yang sudah diubah
train_df.head()

# %% [markdown]
# Membagi fitur pada dataset menjadi dua bagian, yakni numerical features dan categorical features.

# %%
num_cols = train_df.select_dtypes(include=np.number).columns.tolist()
cat_cols = train_df.select_dtypes(include='object').columns.tolist()
print('Numerical columns: ', num_cols)
print('Categorical columns: ', cat_cols)

num_features = num_cols.copy()
cat_features = cat_cols.copy()


num_features.remove('Burn Rate')


# %% [markdown]
# #### a. Categorical Features

# %% [markdown]
# Menghitung jumlah dan persentase karyawan berdasarkan jenis kelamin, jenis perusahaan, fasilitas bekerja dari rumah, dan penunjukan.

# %%
for col in cat_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, data=train_df)
    plt.show()
    count = train_df[col].value_counts()
    percentage = train_df[col].value_counts(normalize=True) * 100
    df = pd.DataFrame({'count': count, 'percentage': percentage.round(2)})
    print(df)
    print()

# %% [markdown]
# #### b. Numerical Features

# %% [markdown]
# Menampilkan distribusi data pada fitur numerik.

# %%
train_df.hist(figsize=(12, 8))
plt.show()

# %% [markdown]
# Berdasarkan grafik histogram di atas, didapatkan distribusi data pada fitur numerik yaitu:
# - `Designation`: Distribusi data normal.
# - `Resource Allocation`: Distribusi data normal.
# - `Mental Fatigue Score`: Distribusi data normal.
# - `Burn Rate`: Distribusi data normal.
# - `Days Employed`: Distribusi data Ragged Plateau.

# %% [markdown]
# ### 2.2.6. Multivariate Analysis

# %% [markdown]
# #### a. Categorical Features

# %% [markdown]
# Mengecek rata-rata `Burn Rate` terhadap masing-masing fitur kategori yaitu `Gender`, `Company Type`, dan `WFH Setup Available` untuk mengetahui pengaruh fitur tersebut terhadap `Burn Rate`.
# 

# %%
for col in cat_cols:
  sns.catplot(x=col, y='Burn Rate', data=train_df, kind='bar', height=4, aspect=2)
  plt.title(f'Rata-rata \'Burn Rate\' terhadap {col}')

# %% [markdown]
# Berdasarkan histogram di atas, diperoleh kesimpulan:
# - Rata-rata `Burn Rate` terhadap `Gender` memberikan sedikit pengaruh di mana rata-rata `Burn Rate` untuk `Gender` `Male` lebih tinggi dibandingkan `Gender` `Female`.
# - `Company Type` tidak memberikan pengaruh terhadap `Burn Rate` karena rata-rata `Burn Rate` untuk `Company Type` `Service` dan `Product` hampir sama.
# - Ketidaktersediaan WFH (Work From Home) membuat `Burn Rate` lebih tinggi  dibandingkan yang tersedia.

# %% [markdown]
# Menghapus fitur `Gender` karena tidak memiliki korelasi terhadap `Burn Rate`.

# %%
train_df.drop('Gender', axis=1, inplace=True)
cat_features.remove('Gender')

train_df.head()

# %% [markdown]
# #### b. Numerical Features

# %% [markdown]
# Mengecek rata-rata `Burn Rate` terhadap masing-masing fitur numerik, yakni `Designation`, `Resource Allocation`, `Mental Fatigue Score`, `Burn Rate`, dan `Days Employed`.

# %%
sns.pairplot(train_df, diag_kind='kde')

# %% [markdown]
# Berdasarkan diagram di atas, dapat disimpulkan:
# - Fitur `Burn Rate` memiliki pola sebaran data dengan korelasi positif terhadap `Designation`, `Resource Allocation`, dan `Mental Fatigue Score`.
# - Fitur `Burn Rate` memiliki pola sebaran data yang tidak beraturan terhadap `Days Employed`, sehingga tidak memiliki korelasi.

# %% [markdown]
# ### 2.2.7. Correlation Matrix

# %% [markdown]
# Pengecekan korelasi atau hubungan antar fitur numerik menggunakan heatmap correlation matrix.

# %%
plt.figure(figsize=(12, 8))
correlation_matrix = train_df.corr(numeric_only=True).round(2)

sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Fitur Numerik', size=16)

# %% [markdown]
# Berdasarkan diagram heatmap di atas, dapat disimpulkan bahwa:
# - Rentang nilai adalah 0 sampai 1.
# - Semakin mendekati 1, semakin kuat positif korelasi antar variabel.
# - Semakin mendekati 0, semakin rendah atau tidak ada korelasi antar variabel.
# - Semakin mendekati -1, semakin kuat negatif korelasi antar variabel.
# - Korelasi antar fitur numerik yang memiliki korelasi kuat positif adalah `Burn Rate` terhadap `Designation`, `Resource Allocation`, dan `Mental Fatigue Score`.
# - Korelasi antar fitur numerik yang tidak memiliki korelasi adalah `Burn Rate` terhadap `Days Employed`.

# %% [markdown]
# Menghapus fitur `Days Employed` karena tidak memiliki korelasi terhadap `Burn Rate`.

# %%
train_df.drop('Days Employed', axis=1, inplace=True)
num_features.remove('Days Employed')

train_df.head()

# %% [markdown]
# # **3. Data Preparation**

# %% [markdown]
# ## 3.1. Encoding Fitur Kategori

# %% [markdown]
# Melakukan proses encoding pada fitur kategori `Gender`, `Company Type`, dan `WFH Setup Available`.

# %%
train_df = pd.concat([train_df, pd.get_dummies(train_df['Company Type'], prefix='Company Type')], axis=1)
train_df = pd.concat([train_df, pd.get_dummies(train_df['WFH Setup Available'], prefix='WFH Setup Available')], axis=1)
train_df.drop(['Company Type', 'WFH Setup Available'], axis=1, inplace=True)
train_df.head()

# %% [markdown]
# ## 3.2. Dataset Split

# %% [markdown]
# Membagi dataset menjadi data latih dan data validasi kemudian menampilkan total dataset, total data latih dan total data validasi.
# 

# %%
from sklearn.model_selection import train_test_split

X = train_df.drop(['Burn Rate'], axis=1)
y = train_df['Burn Rate']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# %%
print(f'Total # of sample in whole dataset: {len(X_train) + len(X_val)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in val dataset: {len(X_val)}')

# %% [markdown]
# ## 3.3. Standarisasi

# %% [markdown]
# Melakukan standarisasi pada fitur numerik yaitu `Designation`, `Resource Allocation`, dan `Mental Fatigue Score` menggunakan StandardScaler untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma.

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train[num_features])
X_train[num_features] = scaler.transform(X_train.loc[:, num_features])
X_train[num_features].head()

# %%
X_train[num_features].describe().round(4)

# %% [markdown]
# # **4. Model Development**

# %% [markdown]
# Mempersiapkan dataframe untuk menganalisis ketiga model yang akan digunakan yaitu Boosting Algorithm, K-Nearest Neighbor (KNN), Random Forest.

# %%
models = pd.DataFrame(index=['train_mse', 'val_mse'],
                      columns=['Boosting', 'KKN', 'RandomForest'])

# %% [markdown]
# ## 4.1. Boosting Algorithm

# %% [markdown]
# Algoritma ini didesain untuk meningkatkan kinerja atau keakuratan prediksi dengan menggabungkan beberapa model sederhana yang dianggap lemah dan membentuk suatu model yang kuat dengan cara menggabungkannya.

# %%
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import AdaBoostRegressor

boosting = AdaBoostRegressor(random_state=42, learning_rate=0.05)
boosting.fit(X_train, y_train)
models.loc['train_mse', 'Boosting'] = mean_squared_error(y_train, boosting.predict(X_train))

# %% [markdown]
# ## 4.2. Algoritma K-Nearest Neighbor (KNN)

# %% [markdown]
# Algoritma KNN menggunakan kesamaan fitur untuk memprediksi nilai dari setiap data baru dengan cara membandingkan jarak satu sampel ke sampel pelatihan lain dan memilih sejumlah K tetangga terdekat (dengan K adalah sebuah angka positif).

# %%
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)

models.loc['train_mse', 'KKN'] = mean_squared_error(y_train, knn.predict(X_train))

# %% [markdown]
# ## 4.3. Algoritma Random Forest

# %% [markdown]
# Algoritma random forest dibangun secara acak dan dianggap sebagai "ensemble learner" yang kuat untuk digunakan dalam melakukan klasifikasi dan regresi, dan dapat digunakan untuk menangani data yang tidak seimbang dan memiliki banyak fitur.

# %%
from sklearn.ensemble import RandomForestRegressor

RF = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=12, n_jobs=1)
RF.fit(X_train, y_train)

models.loc['train_mse', 'RandomForest'] = mean_squared_error(y_train, RF.predict(X_train))

# %% [markdown]
# # **5. Evaluasi Model**

# %% [markdown]
# Melakukan proses scaling fitur numerik pada validation dataset agar skala antara train dataset dan validation dataset sama.

# %%
X_val.loc[:, num_features] = scaler.transform(X_val.loc[:, num_features])

# %% [markdown]
# Melakukan evaluasi model untuk ketiga algoritma yang digunakan yaitu Boosting Algorithm, K-Nearest Neighbor (KNN), Random Forest menggunakan metrik Mean Squared Error (MSE).

# %%
mse = pd.DataFrame(columns=['train', 'val'], index=['Boosting', 'KKN', 'RandomForest'])

model_dict = {'Boosting': boosting, 'KKN': knn, 'RandomForest': RF}

for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_train, model.predict(X_train))
    mse.loc[name, 'val'] = mean_squared_error(y_val, model.predict(X_val))
    
mse

# %%
fig, ax = plt.subplots()
mse.sort_values(by='val', ascending=False).plot(kind='bar', ax=ax, zorder=3)
ax.grid(zorder=0)

# %% [markdown]
# Berdasarkan grafik di atas, dapat disimpulkan bahwa:
# - Model Random Forest memiliki nilai MSE terkecil, yaitu 0.002355 pada data latih dan 0.00313 pada data validasi.
# - Model K-Nearest Neighbor (KNN) memiliki nilai MSE 0.002592 pada data latih dan 0.003174 pada data validasi.
# - Model Boosting Algorithm memiliki nilai MSE terbesar, yaitu 0.003867 pada data latih dan 0.003991 pada data validasi.

# %% [markdown]
# Mencoba prediksi `Burn Rate` menggunakan model yang telah dibuat.

# %%
prediction = X_val.iloc[:1].copy()
pred_dict = {'y_true': y_val.iloc[:1]}
for name, model in model_dict.items():
    pred_dict[name] = model.predict(prediction).round(2)
    
pd.DataFrame(pred_dict)

# %% [markdown]
# Berdasarkan output tabel di atas dapat dilihat bahwa urutan algoritma yang paling mendekati dengan nilai y_true adalah Random Forest. Nilai y_true sebesar 0.63 dan nilai prediksi Random Forest sebesar 0.62.

# %%



