import pandas as pd
import matplotlib.pyplot as plt

# Dataset
file_path = 'C:/Users/Juan David Naranjo/Desktop/ADA School/analisis-y-prediccion/EmployeesData.csv'
df = pd.read_csv(file_path)

missing_values = df.isnull().sum()
print("Valores faltantes:\n", missing_values)

# Convertir la columna 'LeaveOrNot' de valores binarios a etiquetas categóricas
df['LeaveOrNot'] = df['LeaveOrNot'].map({0: 'Not Leave', 1: 'Leave'})
print("Conversión de LeaveOrNot:\n", df['LeaveOrNot'].value_counts())

# Eliminar filas con valores faltantes en las columnas 'ExperienceInCurrentDomain' y 'JoiningYear'
df = df.dropna(subset=['ExperienceInCurrentDomain', 'JoiningYear'])

# Datos faltantes en la columna 'Age' con la media
mean_age = df['Age'].mean()
df['Age'].fillna(mean_age, inplace=True)

# IDatos faltantes en la columna 'PaymentTier' con la moda
mode_payment_tier = df['PaymentTier'].mode()[0]
df['PaymentTier'].fillna(mode_payment_tier, inplace=True)

# Función para eliminar valores atípicos basados en IQR
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Eliminar valores atípicos en 'Age' y 'ExperienceInCurrentDomain'
df = remove_outliers_iqr(df, 'Age')
df = remove_outliers_iqr(df, 'ExperienceInCurrentDomain')

# Graficar la distribución de los sexos con un gráfico de torta
gender_counts = df['Gender'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribución de Géneros')
plt.show()