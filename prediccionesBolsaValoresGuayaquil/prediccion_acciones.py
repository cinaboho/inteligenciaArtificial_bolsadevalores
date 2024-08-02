import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from datetime import datetime

# Cargar el archivo CSV automáticamente
def cargar_datos():
    ruta_archivo = 'stats_BVG_latest.csv'
    df = pd.read_csv(ruta_archivo)
    return df

def preprocesar_datos(df):
    if 'Último Precio' in df.columns:
        closing_price = df['Último Precio']
    else:
        raise KeyError(f"No se encontró la columna 'Último Precio' en el DataFrame. Columnas disponibles: {df.columns.tolist()}")

    scaler = MinMaxScaler(feature_range=(0, 1))
    datos_escalados = scaler.fit_transform(closing_price.values.reshape(-1, 1))
    return datos_escalados, scaler

def crear_conjuntos(datos_escalados, ventana):
    X, y = [], []
    for i in range(ventana, len(datos_escalados)):
        X.append(datos_escalados[i-ventana:i, 0])
        y.append(datos_escalados[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y

def crear_modelo(input_shape):
    modelo = Sequential()
    modelo.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    modelo.add(Dropout(0.2))
    modelo.add(LSTM(units=50, return_sequences=False))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(units=1))
    modelo.compile(optimizer='adam', loss='mean_squared_error')
    return modelo

def entrenar_modelo(modelo, X_train, y_train, epochs=50, batch_size=32):
    modelo.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return modelo

def hacer_predicciones(modelo, X_test, scaler):
    predicciones = modelo.predict(X_test)
    predicciones = scaler.inverse_transform(predicciones)
    return predicciones

def visualizar_resultados(df, predicciones, ventana, fecha_inicio, fecha_fin):
    plt.figure(figsize=(12, 6))

    fechas = pd.to_datetime(df['Fecha'])
    df.set_index('Fecha', inplace=True)

    plt.plot(fechas, df['Último Precio'], color='blue', label='Precio real')

    fechas_predicciones = pd.date_range(start=fecha_inicio, end=fecha_fin, periods=len(predicciones))
    plt.plot(fechas_predicciones, predicciones, color='red', label='Predicciones')

    plt.title('Predicción de precios de acciones')
    plt.xlabel('Fecha')
    plt.ylabel('Precio de las acciones')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def procesar_datos():
    fecha_inicio = fecha_inicio_entry.get()
    fecha_fin = fecha_fin_entry.get()

    try:
        # Convertir las fechas de entrada a datetime para validación
        datetime.strptime(fecha_inicio, "%Y-%m-%d")
        datetime.strptime(fecha_fin, "%Y-%m-%d")
    except ValueError:
        messagebox.showerror("Error", "El formato de fecha debe ser YYYY-MM-DD.")
        return

    df = cargar_datos()
    df['Fecha'] = pd.to_datetime(df['Fecha'])  # Asegurarse de que 'Fecha' esté en formato datetime

    # Filtrar los datos según el rango de fechas seleccionado
    df = df[(df['Fecha'] >= fecha_inicio) & (df['Fecha'] <= fecha_fin)]

    if df.empty:
        messagebox.showerror("Error", "No hay datos en el rango de fechas seleccionado.")
        return

    ventana = 60  # Tamaño de ventana fijo

    datos_escalados, scaler = preprocesar_datos(df)
    X, y = crear_conjuntos(datos_escalados, ventana)

    X_train, X_test = X[:int(X.shape[0]*0.8)], X[int(X.shape[0]*0.8):]
    y_train, y_test = y[:int(y.shape[0]*0.8)], y[int(y.shape[0]*0.8):]

    modelo = crear_modelo((X_train.shape[1], 1))
    modelo = entrenar_modelo(modelo, X_train, y_train, epochs=50, batch_size=32)

    predicciones = hacer_predicciones(modelo, X_test, scaler)
    visualizar_resultados(df, predicciones, ventana, fecha_inicio, fecha_fin)

# Configuración de la interfaz gráfica
root = tk.Tk()
root.title("Predicción de Precios de Acciones")

tk.Label(root, text="Fecha de inicio (YYYY-MM-DD):").pack(pady=5)
fecha_inicio_entry = tk.Entry(root)
fecha_inicio_entry.pack(pady=5)
fecha_inicio_entry.insert(0, "2022-01-01")

tk.Label(root, text="Fecha de fin (YYYY-MM-DD):").pack(pady=5)
fecha_fin_entry = tk.Entry(root)
fecha_fin_entry.pack(pady=5)
fecha_fin_entry.insert(0, "2022-12-31")

tk.Button(root, text="Predecir", command=procesar_datos).pack(pady=20)

root.geometry("400x200")  # Tamaño fijo de la ventana
root.mainloop()
