import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from Limpieza_Datos import df_limpio

X = df_limpio[['Dorms', 'Baths', 'Built Area', 'Total Area', 'Comuna']] # Variables independientes
X = pd.get_dummies(X, columns=['Comuna'], drop_first=True) # Convierte la columna 'Comuna' en variables dummy
Y = df_limpio['Price_UF'] # Variable dependiente

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) # División de datos en entrenamiento y prueba, 80% - 20%
modelo_random_forest = RandomForestRegressor(n_estimators=100, random_state=42) # Crear el modelo de Random Forest
modelo_random_forest.fit(X_train, Y_train) # Entrenar el modelo con los datos de entrenamiento

predicciones = modelo_random_forest.predict(X_test) # Hacer predicciones con los datos de prueba
importancia_caracteristicas = modelo_random_forest.feature_importances_ # Obtener la importancia de cada característica
caracteristicas = X.columns

importancia_df = pd.DataFrame({'Características': caracteristicas, 'Importancia': importancia_caracteristicas}) # Crear un DataFrame con las características y su importancia
importancia_df = importancia_df.sort_values(by='Importancia', ascending=False) # Ordenar el DataFrame por importancia

top = importancia_df.head(10) # Seleccionar las 10 características más influyentes

mae = mean_absolute_error(Y_test, predicciones) # Calcular el error absoluto medio en los datos de prueba
r2 = r2_score(Y_test, predicciones) # Calcular el coeficiente de determinación R^2

# Resultados
print(f'El error absoluto medio del modelo fue: {mae:.1f} UF')
print(f'El coeficiente de determinación del modelo fue: {r2:.2f} UF')

# Gráfico de las 10 características más influyentes
plt.figure(figsize=(10, 6))
sns.barplot(x='Importancia', y='Características', data=top, color='Orange')
plt.title('Características más influyentes en el modelo de Random Forest')
plt.show()
