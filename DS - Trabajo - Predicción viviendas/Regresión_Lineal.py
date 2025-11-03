import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from Limpieza_Datos import df_limpio

X = df_limpio[['Dorms', 'Baths', 'Built Area', 'Total Area', 'Parking']] 
Y = df_limpio['Price_UF'] 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
predictor_prices = LinearRegression()
predictor_prices.fit(X_train, Y_train) 

mae = mean_absolute_error(Y_test, predictor_prices.predict(X_test)) # Calcular el error absoluto medio en los datos de prueba
r2 = r2_score(Y_test, predictor_prices.predict(X_test)) # Calcular el coeficiente de determinación R^2

print(f'Error absoluto medio: {mae}')
print(f'R^2: {r2}')

plt.figure(figsize=(10,6))
sns.scatterplot(x=Y_test, y=predictor_prices.predict(X_test), color='red', alpha=0.6)
plt.xlabel('Precio Real')
plt.ylabel('Precio Predicho')
plt.title('Comparación de Precios Reales vs Predichos')
plt.show()

