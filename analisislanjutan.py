# =========================================
# ANALISIS DATA E-COMMERCE
# =========================================

# =========================================
# 1. IMPORT LIBRARY
# =========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# =========================================
# 2. MEMBACA DATA
# =========================================

df = pd.read_csv('DataAnalisis.csv')

print("===== DATA AWAL =====")
print(df.head())

# =========================================
# 3. CEK INFORMASI DATA
# =========================================

print("\n===== INFO DATA =====")
print(df.info())

# =========================================
# 4. CEK DATA KOSONG
# =========================================

print("\n===== DATA KOSONG =====")
print(df.isnull().sum())

# =========================================
# 5. MEMBERSIHKAN DATA
# =========================================

df = df.dropna()

print("\n===== DATA SETELAH CLEANING =====")
print(df.isnull().sum())

# =========================================
# 6. MENGUBAH FORMAT TANGGAL
# =========================================

df['Order_Date'] = pd.to_datetime(df['Order_Date'])

# =========================================
# 7. ANALISIS TREN PENJUALAN BULANAN
# =========================================

df['Month'] = df['Order_Date'].dt.to_period('M').astype(str)

monthly_sales = df.groupby('Month')['Total_Sales'].sum()

plt.figure(figsize=(10,5))

plt.plot(
    monthly_sales.index,
    monthly_sales.values,
    marker='o'
)

plt.title('Tren Penjualan Bulanan')
plt.xlabel('Bulan')
plt.ylabel('Total Penjualan')

plt.xticks(rotation=45)

plt.show()

# =========================================
# 8. ANALISIS KORELASI
# =========================================

correlation = df[['Total_Sales', 'Ad_Budget']].corr()

plt.figure(figsize=(6,4))

sns.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm'
)

plt.title('Korelasi Budget Iklan dan Penjualan')

plt.show()

# =========================================
# 9. ANALISIS PRODUK TERLARIS
# =========================================

category_sales = df.groupby('Product_Category')['Total_Sales'].sum()

plt.figure(figsize=(8,5))

category_sales.plot(kind='bar')

plt.title('Penjualan Berdasarkan Kategori Produk')
plt.xlabel('Kategori Produk')
plt.ylabel('Total Penjualan')

plt.show()

# =========================================
# 10. IDENTIFIKASI PRODUK UNDERPERFORMER
# =========================================

avg_price = df['Price_Per_Unit'].mean()

underperformer = df[
    (df['Price_Per_Unit'] > avg_price) &
    (df['Quantity'] <= 2)
]

print("\n===== PRODUK UNDERPERFORMER =====")
print(underperformer)

plt.figure(figsize=(8,5))

plt.scatter(
    df['Price_Per_Unit'],
    df['Quantity']
)

plt.title('Harga vs Quantity')
plt.xlabel('Price Per Unit')
plt.ylabel('Quantity')

plt.show()

# =========================================
# 11. RFM ANALYSIS
# =========================================

snapshot_date = df['Order_Date'].max() + dt.timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'Order_Date': lambda x: (snapshot_date - x.max()).days,
    'Order_ID': 'count',
    'Total_Sales': 'sum'
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

print("\n===== RFM ANALYSIS =====")
print(rfm.head())

# =========================================
# 12. MEMBERIKAN SKOR RFM
# =========================================

rfm['R_Score'] = pd.qcut(
    rfm['Recency'],
    5,
    labels=[5,4,3,2,1]
)

rfm['F_Score'] = pd.qcut(
    rfm['Frequency'].rank(method='first'),
    5,
    labels=[1,2,3,4,5]
)

rfm['M_Score'] = pd.qcut(
    rfm['Monetary'],
    5,
    labels=[1,2,3,4,5]
)

rfm['RFM_Group'] = (
    rfm['R_Score'].astype(str) +
    rfm['F_Score'].astype(str) +
    rfm['M_Score'].astype(str)
)

print("\n===== SKOR RFM =====")
print(rfm.head())

# =========================================
# 13. ANALISIS EFISIENSI KATEGORI
# =========================================

category_analysis = df.groupby('Product_Category').agg({
    'Total_Sales': 'sum',
    'Ad_Budget': 'sum'
})

category_analysis['Efficiency'] = (
    category_analysis['Total_Sales'] /
    category_analysis['Ad_Budget']
)

print("\n===== EFISIENSI KATEGORI =====")
print(category_analysis)

category_analysis = category_analysis.sort_values(
    by='Efficiency'
)

plt.figure(figsize=(8,5))

category_analysis['Efficiency'].plot(
    kind='barh'
)

plt.title('Efisiensi Kategori Produk')
plt.xlabel('Efficiency')

plt.show()

# =========================================
# 14. UJI HIPOTESIS IKLAN
# =========================================

median_ad = df['Ad_Budget'].median()

high_ads = df[df['Ad_Budget'] > median_ad]

low_ads = df[df['Ad_Budget'] <= median_ad]

high_mean = high_ads['Total_Sales'].mean()

low_mean = low_ads['Total_Sales'].mean()

print("\n===== UJI HIPOTESIS =====")

print(
    "Rata-rata penjualan iklan tinggi:",
    high_mean
)

print(
    "Rata-rata penjualan iklan rendah:",
    low_mean
)

# =========================================
# 15. REGRESI LINEAR
# =========================================

X = df[['Ad_Budget']]

y = df['Total_Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

print("\n===== REGRESI LINEAR =====")

print(
    "Koefisien Iklan:",
    model.coef_[0]
)

print(
    "Akurasi Model (R2 Score):",
    model.score(X_test, y_test)
)

# =========================================
# 16. KESIMPULAN AKHIR
# =========================================

print("\n===== KESIMPULAN =====")

print("""
1. Penjualan berubah setiap bulan.

2. Budget iklan memiliki hubungan dengan total penjualan.

3. Terdapat produk underperformer:
   harga tinggi namun quantity rendah.

4. RFM Analysis membantu menentukan pelanggan loyal.

5. Efisiensi kategori menunjukkan
   kategori paling efektif dalam penggunaan iklan.

6. Regresi linear digunakan untuk
   memprediksi penjualan berdasarkan budget iklan.
""")