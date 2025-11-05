import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from Limpieza_Datos import df_limpio

X = df_limpio[['Dorms', 'Comuna', 'Baths', 'Built Area', 'Total Area']]
X = pd.get_dummies(X, columns=['Comuna'], drop_first=True) # Convierte la columna 'Comuna' en variables dummy
Y = df_limpio['Price_UF']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) # División de datos en entrenamiento y prueba, 80% - 20%
modelo_regresion_lineal = LinearRegression() 
modelo_regresion_lineal.fit(X_train, Y_train) 

correlaciones = df_limpio[['Price_UF', 'Dorms', 'Baths', 'Built Area', 'Total Area']].corr() # Calcular la matriz de correlación
predicciones = modelo_regresion_lineal.predict(X_test) # Hacer predicciones con los datos de prueba
mae = mean_absolute_error(Y_test, modelo_regresion_lineal.predict(X_test)) # Calcular el error absoluto medio en los datos de prueba
r2 = r2_score(Y_test, modelo_regresion_lineal.predict(X_test)) # Calcular el coeficiente de determinación R^2

# Resultados
print(f'Matriz de correlación: \n{correlaciones}\n')
print(f'El error absoluto medio del modelo fue: {mae:.1f} UF')
print(f'El coeficiente de determinación del modelo fue: {r2:.2f} UF')

# Gráfico de la matriz de correlación
plt.figure(figsize=(10, 6))
sns.heatmap(correlaciones, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de correlación')
plt.show()

# Gráfico de dispersión de valores reales vs predicciones, con recta de referencia
plt.figure(figsize=(10, 6))
sns.scatterplot(x=Y_test, y=predicciones, color = 'Blue', alpha=0.6)
plt.xlabel('Valores reales')
plt.ylabel('Predicciones')
plt.title('Comparación de valores reales y predicciones (UF)')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color='red', linestyle='--')
plt.show()


