import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('nilai_siswa.csv')
print(data.info()) 
print(data.head()) 
print(data.describe())

print("Rata rata:", data['Nilai'].mean())
print("Media:", data['Nilai'].median())
print("Mode:", data['Nilai'].mode())

# matematika = data[data['Mata Pelajaran'] == 'matematika']
# print(matematika)

# bahasa_indonesia = data[data['Mata Pelajaran'] == 'Bahasa Indonesia']
# print(bahasa_indonesia)

# bahasa_inggris = data[data['Mata Pelajaran'] == 'Bahasa Inggris']
# print(bahasa_inggris)

# fisika = data[data['Mata Pelajaran'] == 'Fisika']
# print(fisika) 

# produktif = data[data['Mata Pelajaran'] == 'Produktif']
# print(produktif)

data.groupby('Matpel')['Nilai'].agg(['max','min'])

rata = data.groupby('Matpel')['Nilai'].mean()
rata.plot(kind='bar', color=['blue', 'orange', 'green', 'red', 'purple'])
plt.title('Rata-Rata Nilai per Matpel')
plt.xlabel('Mata Pelajaran')
plt.ylabel('Nilai Rata-Rata')
plt.show()

sns.boxplot(x='Matpel', y='Nilai', data=data)
plt.title('Sebaran Nilai per Mata Pelajaran')
plt.tight_layout()
plt.show()
  