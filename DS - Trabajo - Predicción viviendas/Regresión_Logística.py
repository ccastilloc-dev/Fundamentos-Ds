import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from Limpieza_Datos import df_limpio

mediana = df_limpio['Price_UF'].median()
df_limpio['Categoria_Por_Precio'] = np.where(df_limpio['Price_UF'] > mediana, 'Costosa', 'Económica') # Se crean categorías basadas en la mediana del precio

X = df_limpio[['Dorms', 'Comuna', 'Baths', 'Built Area', 'Total Area']] # Variables independientes
X = pd.get_dummies(X, columns=['Comuna'], drop_first=True) # Convierte la columna 'Comuna' en variables dummy
Y = df_limpio['Categoria_Por_Precio'] # Variable dependiente
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) # División de datos en entrenamiento y prueba, 80% - 20%
modelo_regresion_logistica = LogisticRegression(max_iter=1500) # Crear el modelo de regresión logística
modelo_regresion_logistica.fit(X_train, Y_train) # Entrenar el modelo con los datos de entrenamiento

predicciones = modelo_regresion_logistica.predict(X_test) # Hacer predicciones con los datos de prueba
exactitud = accuracy_score(Y_test, predicciones) # Calcular la exactitud del modelo
precision = precision_score(Y_test, predicciones, pos_label='Costosa') # Calcular la precisión del modelo
recall = recall_score(Y_test, predicciones, pos_label='Costosa') # Calcular el recall del modelo
matriz_confusion = confusion_matrix(Y_test, predicciones) # Calcular la matriz de confusión

# Resultados
print(f'La exactitud del modelo (Porcentaje de predicciones correctas) fue: {exactitud:.1f}')
print(f'La precisión del modelo (Porcentaje de verdaderos positivos sobre el total de positivos predichos) fue: {precision:.1f}')
print(f'La sensibilidad del modelo (Porcentaje de verdaderos positivos sobre el total de positivos reales) fue: {recall:.1f}')
print(f'Matriz de Confusión:\n{matriz_confusion}')

# Gráfico de la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Oranges', xticklabels=['Económica', 'Costosa'], yticklabels=['Económica', 'Costosa'])
plt.xlabel('Predicción')
plt.ylabel('Realidad')
plt.title('Matriz de Confusión')
plt.show()

