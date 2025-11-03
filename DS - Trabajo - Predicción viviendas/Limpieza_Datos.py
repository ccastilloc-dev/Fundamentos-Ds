import pandas as pd
import numpy as np

df_viviendas = pd.read_csv('2023-03-08 Precios Casas RM.csv')
categorias_importantes = ['Price_UF', 'Comuna', 'Dorms', 'Baths', 'Built Area', 'Total Area', 'Parking']


df_limpio = df_viviendas.dropna(subset=categorias_importantes).copy() # Crea un DataFrame sin filas que tienen valores nulos en las columnas importantes
df_sucio = df_viviendas[df_viviendas[categorias_importantes].isnull().any(axis=1)].copy() # Crea un DataFrame con filas que tienen valores nulos en las columnas importantes
df_limpio.drop_duplicates(keep='first', inplace=True) # Elimina filas duplicadas, manteniendo la primera aparición
df_limpio.reset_index(drop=True, inplace=True) # Reinicia los índices del DataFrame limpio
df_limpio.drop(columns=['Price_CLP', 'Price_USD', 'Ubicacion', 'id', 'Realtor'], inplace=True) # Elimina columnas innecesarias

print(df_limpio.info())
print(df_limpio.head())













