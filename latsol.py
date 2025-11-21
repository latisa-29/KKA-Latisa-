import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('nilai_siswa.csv')
print(data.info())
print(data.head())
print(data.describe())

print("rata Rata:", data['Nilai'].mean())
print("Median:", data['Nilai'].median())
print("Modus:", data['Nilai'].mode()[0])

matematika = data[data['Matpel'] == 'Matematika']
print(matematika)

fisika = data[data['Matpel'] == 'Fisika']
print(fisika)

produktif = data[data['Matpel'] == 'Produktif']
print(produktif)

Indonesia = data[data['Matpel'] == 'Bahasa Indonesia']
print(Indonesia)

Inggris = data[data['Matpel'] == 'Bahasa Inggris']
print(Inggris)

data.groupby('Matpel')['Nilai'].agg(['max','min'])

rata = data.groupby('Matpel')['Nilai'].mean()
rata.plot(kind='bar', color=['blue', 'orange', 'green', 'red', 'purple'])
plt.title('Rata-Rata Nilai per Matpel')
plt.xlabel('Mata Pelajaran')
plt.ylabel('Nilai Rata-Rata')
plt.show()

sns.boxplot(x='Mapel', y='Nilai', data=data)
plt.title('Sebaran Nilai per Mata Pelajaran')
plt.tight_layout()
plt.show()