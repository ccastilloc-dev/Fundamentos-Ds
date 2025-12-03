
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler

from Limpieza_Datos import df_limpio  #Usa el DataFrame limpio

df_limpio['Price_per_m2'] = df_limpio['Price_UF'] / (df_limpio['Built Area'].replace(0, np.nan))
df_limpio['Baths_per_Dorm'] = df_limpio['Baths'] / (df_limpio['Dorms'].replace(0, np.nan))

X = df_limpio[['Dorms', 'Baths', 'Built Area', 'Total Area', 'Price_per_m2', 'Baths_per_Dorm', 'Comuna']]
X = pd.get_dummies(X, columns=['Comuna'], drop_first=True)
y = df_limpio['Price_UF']

q_low = y.quantile(0.01)
q_high = y.quantile(0.99)
mask = (y >= q_low) & (y <= q_high)
X = X[mask]
y = y[mask]

y_log = np.log1p(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential([
    Dense(256, activation='relu', kernel_regularizer=l2(0.001), input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
history = model.fit(X_train_scaled, y_train, validation_split=0.2,
                    epochs=500, batch_size=32, verbose=1,
                    callbacks=[early_stop])

predictions_log = model.predict(X_test_scaled).flatten()
predictions = np.expm1(predictions_log) 
y_test_original = np.expm1(y_test)

mae = mean_absolute_error(y_test_original, predictions)
r2 = r2_score(y_test_original, predictions)

print(f"MAE del modelo final: {mae:.2f}")
print(f"R² del modelo final: {r2:.2f}")

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Pérdida (MSE) - Entrenamiento', color='blue')
plt.plot(history.history['val_loss'], label='Pérdida (MSE) - Validación', color='orange')
plt.title('Curva de pérdida (MSE) - Entrenamiento vs Validación')
plt.xlabel('Épocas')
plt.ylabel('MSE')
plt.yscale('log') 
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test_original, y=predictions, alpha=0.6)

#Línea de referencia (y = x)
plt.plot([y_test_original.min(), y_test_original.max()],
         [y_test_original.min(), y_test_original.max()],
         color='red', linestyle='--', label='Referencia y=x')

#Añadir métricas en el gráfico
plt.text(0.05, 0.95, f'R²={r2:.2f}\nMAE={mae:.0f} UF',
         transform=plt.gca().transAxes, fontsize=12,
         bbox=dict(facecolor='white', alpha=0.7))

plt.xlabel('Valores reales (Price_UF)')
plt.ylabel('Predicciones')
plt.title('Comparación entre valores reales y predicciones')
plt.legend()
plt.grid(True)
plt.show()

model.save('modelo_red_neuronal_final.h5')
print("Modelo guardado como 'modelo_red_neuronal_final.h5'")

def load_and_predict(model_path, data_dict):
    loaded_model = load_model(model_path)
    new_df = pd.DataFrame([data_dict])
    new_df['Price_per_m2'] = new_df['Built Area'] / (new_df['Built Area'].replace(0, np.nan))
    new_df['Baths_per_Dorm'] = new_df['Baths'] / (new_df['Dorms'].replace(0, np.nan))
    new_df = pd.get_dummies(new_df, columns=['Comuna'])
    new_df = new_df.reindex(columns=X.columns, fill_value=0)
    new_scaled = scaler.transform(new_df)
    pred_log = loaded_model.predict(new_scaled).flatten()[0]
    return np.expm1(pred_log)

#Ejemplo de uso
nuevo_dato = {
    'Dorms': 3,
    'Baths': 2,
    'Built Area': 120,
    'Total Area': 200,
    'Comuna': 'Santiago'
}
print(f"Predicción con modelo cargado: {load_and_predict('modelo_red_neuronal_final.h5', nuevo_dato):.2f} UF")
