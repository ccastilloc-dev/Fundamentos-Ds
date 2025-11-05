import pandas as pd
import numpy as np

df_viviendas = pd.read_csv('2023-03-08 Precios Casas RM.csv')
categorias_importantes = ['Price_UF', 'Comuna', 'Dorms', 'Baths', 'Built Area', 'Total Area']


df_limpio = df_viviendas.dropna(subset=categorias_importantes).copy() # Crea un DataFrame sin filas que tienen valores nulos en las columnas importantes
df_sucio = df_viviendas[df_viviendas.isnull().any(axis=1)].copy() # Crea un DataFrame con filas que tienen valores nulos en alguna columna
df_limpio.drop_duplicates(keep='first', inplace=True) # Elimina filas duplicadas, manteniendo la primera aparición
df_limpio.drop(columns=['Price_CLP', 'Price_USD', 'Ubicacion', 'id', 'Realtor'], inplace=True) # Elimina columnas innecesarias

####################################################
# Eliminación de outliers usando el método del rango intercuartílico (IQR)
# Exceptuando Dorms y Baths, cuya cantidad mínima se establece en 1
####################################################
Q1 = df_limpio['Price_UF'].quantile(0.25) 
Q3 = df_limpio['Price_UF'].quantile(0.75) 
IQR = Q3 - Q1 
df_limpio = df_limpio[(df_limpio['Price_UF'] >= Q1 - 1.5 * IQR) & (df_limpio['Price_UF'] <= Q3 + 1.5 * IQR)] 

Q1 = df_limpio['Dorms'].quantile(0.25) 
Q3 = df_limpio['Dorms'].quantile(0.75) 
IQR = Q3 - Q1 
df_limpio = df_limpio[(df_limpio['Dorms'] >= 1) & (df_limpio['Dorms'] <= Q3 + 1.5 * IQR)]

Q1 = df_limpio['Baths'].quantile(0.25)
Q3 = df_limpio['Baths'].quantile(0.75) 
IQR = Q3 - Q1
df_limpio = df_limpio[(df_limpio['Baths'] >= 1) & (df_limpio['Baths'] <= Q3 + 1.5 * IQR)]

Q1 = df_limpio['Built Area'].quantile(0.25) 
Q3 = df_limpio['Built Area'].quantile(0.75) 
IQR = Q3 - Q1 
df_limpio = df_limpio[(df_limpio['Built Area'] >= Q1 - 1.5 * IQR) & (df_limpio['Built Area'] <= Q3 + 1.5 * IQR)]

Q1 = df_limpio['Total Area'].quantile(0.25) 
Q3 = df_limpio['Total Area'].quantile(0.75) 
IQR = Q3 - Q1 
df_limpio = df_limpio[(df_limpio['Total Area'] >= Q1 - 1.5 * IQR) & (df_limpio['Total Area'] <= Q3 + 1.5 * IQR)] 

df_limpio.reset_index(drop=True, inplace=True) # Reinicia los índices del DataFrame limpio después del filtrado
df_limpio.to_csv('Precios Casas - Datos Limpios.csv', index=False) # Guarda el DataFrame limpio en un archivo CSV
df_sucio.to_csv('Precios Casas - Datos Sucios.csv', index=False) # Guarda el DataFrame con datos sucios en un archivo CSV














