import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Membaca data
df = pd.read_csv('DataAnalisis.csv')

# Menampilkan data awal
print(df.head())

# Cek data kosong
print(df.isnull().sum())

# Hapus data kosong
df = df.dropna()

# Ubah format tanggal
df['Order_Date'] = pd.to_datetime(df['Order_Date'])

# =========================
# ANALISIS PENJUALAN BULANAN
# =========================

df['Month'] = df['Order_Date'].dt.to_period('M').astype(str)

monthly_sales = df.groupby('Month')['Total_Sales'].sum()

plt.figure(figsize=(10,5))
plt.plot(monthly_sales.index, monthly_sales.values, marker='o')

plt.title('Tren Penjualan Bulanan')
plt.xlabel('Bulan')
plt.ylabel('Total Penjualan')

plt.xticks(rotation=45)

plt.show()

# =========================
# ANALISIS KORELASI
# =========================

correlation = df[['Total_Sales', 'Ad_Budget']].corr()

sns.heatmap(correlation, annot=True, cmap='coolwarm')

plt.title('Korelasi Budget Iklan dan Penjualan')

plt.show()

# =========================
# PRODUK TERLARIS
# =========================

category_sales = df.groupby('Product_Category')['Total_Sales'].sum()

plt.figure(figsize=(8,5))

category_sales.plot(kind='bar')

plt.title('Penjualan Berdasarkan Kategori Produk')
plt.xlabel('Kategori Produk')
plt.ylabel('Total Penjualan')

plt.show()