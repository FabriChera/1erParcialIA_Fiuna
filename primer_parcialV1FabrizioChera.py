import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Función para calcular los coeficientes de regresión manualmente
def regresion_manual(X, y):
    # Agregar una columna de unos para el término independiente
    X = np.c_[np.ones((X.shape[0], 1)), X]
    
    # Calcular los coeficientes utilizando la fórmula de la pseudo inversa
    coeficientes = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    return coeficientes

# Función para predecir los valores de y
def predecir(X, coeficientes):
    Xm = np.c_[np.ones((X.shape[0], 1)), X]
    
    return  Xm @ coeficientes

# Calcular métricas de evaluación manualmente
def rmse(y_true, y_pred):
    error = y_true - y_pred
    return np.sqrt(np.mean((error) ** 2))

def r2F(y_true, y_pred):
   # https://es.wikipedia.org/wiki/Coeficiente_de_determinaci%C3%B3n
   # Indagando un poco existe una funcion en sklearn que devuelve el coeficiente de determinacion
   
    numerador = ((y_true - y_pred) ** 2).sum()
    denominador = ((y_true - y_true.mean()) ** 2).sum()
    r2_1 = 1 - (numerador / denominador)
    r2_2 = r2_score(y_true,y_pred)
    #print(r2_1)
    #print(r2_2)

    return 1 - (numerador / denominador)

# Función para ajustar el modelo y evaluarlo
def ajustar_evaluar_modelo(X, y):
    coeficientes = regresion_manual(X, y)
    y_pred = predecir(X, coeficientes)
    r2_ =[r2F(y,y_pred)]#completar
    rmse_val = [rmse(y, y_pred)]#completar
    return coeficientes, y_pred, r2_, rmse_val


opcion=int(input())
# Cargar los datos
data = pd.read_csv('Mediciones.csv')

# Definir las columnas de características (X) y la columna de objetivo (y)
if opcion==1:
    #imprimir numero de filas y numero de columnas
    print("Número de filas y columnas:", data.shape)
    
    #seleccionar las caracteristicas(variables dependientes) y el objetivo
    caracteristicas = data.columns[0:7 and 9] #[completar]
    objetivo = data.columns[7] #
    
    print(caracteristicas)
    print(objetivo)
elif opcion==2: 
    # modelo completo solo con VTI_F, completar la funcion regresion manual
    
    X = data['VTI_F']
    y = data['Pasos']
    X = X.drop(24)
    y = y.drop(24)
    coef = [regresion_manual(X, y)]# regresion_manual(X, y)
    print(coef)

elif opcion==3: 
    # modelo completo solo con VTI_F, completar las funciones que definen las métricas
    X = data['VTI_F']
    X = X.drop(24)
    y = data['Pasos']
    y = y.drop(24)
    coef = regresion_manual(X, y)
    print(coef)
    y_pred = predecir(X,coef)
    r2_ = r2F(y, y_pred)
    rmse_val = rmse(y, y_pred)
    # imprimir los primeros 2 elementos de y e y_pred
    #  print(y[:3],  y_pred [COMPLETAR])
    #print(y_pred)
    print(y[:3],  y_pred [:3])
    # imprimir r2 y rmse
    print(r2_,  rmse_val )
elif opcion==4: 
    # modelo completo solo con VTI_F, completar la función ajustar_evaluar_modelo
    X_todo = data['VTI_F']  #data[completar]
    y = data['Pasos'] # data[completar]
    X_todo = X_todo.drop(24)
    y = y.drop(24)
    coeficientes_todo, y_pred_todo, r2_todo, rmse_todo = ajustar_evaluar_modelo(X_todo, y)
    print(r2_todo, rmse_todo)
elif opcion==5:
   # Completar la combinaciones de características de los modelos solicitados 
    models = {
        'Modelo_1': ['VTI_F'],
        'Modelo_2': ['VTI_F', 'BPM'],
        'Modelo_3': ['VTI_F','PEEP'],
        'Modelo_4': ['VTI_F','PEEP','BPM'],
        'Modelo_5': ['VTI_F','PEEP','BPM','VTE_F']
      #COMPLETAR EL DICCIONARIO
    }
    for nombre_modelo, lista_caracteristicas in models.items():
        X = data[lista_caracteristicas]#data[completar]
        y = data['Pasos']
        X = X.drop(24)
        y = y.drop(24)
        coeficientes, y_pred, r2, rmse_val = ajustar_evaluar_modelo(X, y)
        print(nombre_modelo,r2, rmse_val)
elif opcion==6:
    # Modelos para cada combinación de PEEP y BPM
    data = data.drop(24)
    valores_peep_unicos = data['PEEP'].unique()#completar sugerencia, utilizar unique()
    valores_bpm_unicos = data['BPM'].unique() #completar
    type(valores_bpm_unicos)
    print(valores_peep_unicos)
    print(valores_bpm_unicos)
    predicciones_totales = []
    for peep in valores_peep_unicos:
        for bpm in valores_bpm_unicos:

            datos_subset = data[(data['PEEP'] == peep) & (data['BPM'] == bpm)] #completar el filtrado de datos, se deben filtrar los datos para cada para par de PEEP y BPM

            X_subset = datos_subset[['VTI_F']]
            y_subset = datos_subset['Pasos']
            coeficientes_subset, y_pred_subset, r2_subset, rmse_subset = ajustar_evaluar_modelo(X_subset, y_subset)
            print(peep, bpm, r2_subset, rmse_subset)
            predicciones_totales.append(y_pred_subset)
    predicciones_concatenadas = np.concatenate(predicciones_totales)
    y=data['Pasos']
    r2_global = r2F(y, predicciones_concatenadas)
    rmse_global = rmse(y, predicciones_concatenadas)
    print('Global', r2_global, rmse_global)
    #print(predicciones_concatenadas)
    #print(data['VTI_F'])
    # Graficar los resultados obtenidos
    plt.scatter(data['VTI_F'][:6],predicciones_concatenadas[0:6], s=20, c='red', alpha=1.0)
    plt.scatter(data['VTI_F'][6:12],predicciones_concatenadas[6:12], s=20, c='green', alpha=1.0)
    plt.scatter(data['VTI_F'][12:18],predicciones_concatenadas[12:18], s=20, c='black', alpha=1.0)
    plt.scatter(data['VTI_F'][18:24],predicciones_concatenadas[18:24], s=20, c='purple', alpha=1.0)
    plt.scatter(data['VTI_F'],data['Pasos'], s=10, c='orange', alpha=1.0)
    plt.text(200, 30000, 'PEEP=0 BPM=12 (Prediccion)', fontsize=10, color='red')
    plt.text(200, 29000, 'PEEP=0 BPM=20 (Prediccion)', fontsize=10, color='green')
    plt.text(200, 28000, 'PEEP=10 BPM=12 (Prediccion)', fontsize=10, color='black')
    plt.text(200, 27000, 'PEEP=10 BPM=20 (Prediccion)', fontsize=10, color='purple')
    plt.text(200, 26000, 'Valores Reales', fontsize=10, color='orange')
    plt.xlabel('VTI Fluke')  # Nombre del eje x
    plt.ylabel('Pasos')  # Nombre del eje y
    plt.grid(True)  # Mostrar cuadrícula 
    plt.show()