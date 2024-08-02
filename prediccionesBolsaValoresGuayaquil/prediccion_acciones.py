import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# drive.mount('/content/drive')

def cargar_datos(ruta_archivo):
    df = pd.read_csv(ruta_archivo)
    return df

def preprocesar_datos(df):
    # Verificar si la columna 'Último Precio' existe en el DataFrame
    if 'Último Precio':
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

def visualizar_resultados(df, predicciones, ventana):
    plt.plot(df['Último Precio'].values, color='blue', label='Precio real')
    plt.plot(range(ventana, len(predicciones)+ventana), predicciones, color='red', label='Predicciones')
    plt.title('Predicción de precios de acciones')
    plt.xlabel('Tiempo')
    plt.ylabel('Precio de las acciones')
    plt.legend()
    plt.show()

def main():
    ruta_archivo = 'stats_BVG_latest.csv'
    ventana = 60

    df = cargar_datos(ruta_archivo)
    datos_escalados, scaler = preprocesar_datos(df)
    X, y = crear_conjuntos(datos_escalados, ventana)

    X_train, X_test = X[:int(X.shape[0]*0.8)], X[int(X.shape[0]*0.8):]
    y_train, y_test = y[:int(y.shape[0]*0.8)], y[int(y.shape[0]*0.8):]

    modelo = crear_modelo((X_train.shape[1], 1))
    modelo = entrenar_modelo(modelo, X_train, y_train, epochs=50, batch_size=32)

    predicciones = hacer_predicciones(modelo, X_test, scaler)
    visualizar_resultados(df, predicciones, ventana)

if __name__ == '__main__':
    main()
