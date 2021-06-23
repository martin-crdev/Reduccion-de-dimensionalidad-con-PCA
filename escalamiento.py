import timeit #Libreria para medir el tiempo de ejecucion de los modelos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model #Funciones para traer datasets y hacer una regresion lineal sencilla

def run():

    X, y = datasets.load_diabetes(return_X_y=True) #Cargamos el dataset
    raw = X[:, None, 2] #Agrego una transformacion en las dimensiones para que se ajuste al formato de entrada

    #Reglas de escalamiento max y min
    max_raw = max(raw)
    min_raw = min(raw)
    scaled = (2*raw - max_raw - min_raw)/(max_raw + min_raw)

    #normalizacion Z-score
    prom = np.average(raw)
    desviacion = np.std(raw)
    zscore = (raw - prom) / desviacion
    

    #La figura contendra nuestros graficos y axs nos define donde los voy a ubicar
    fig, axs = plt.subplots(3, 1, sharex = True) #Sharex indica que compartan el eje de las x

    axs[0].hist(raw) #Asignamos un histograma con los datos raw en el arreglo de los ejes 0
    axs[1].hist(scaled) #Asignamos un histogramas con los datos scaled
    axs[2].hist(zscore)
    plt.show()

    #Modelos para entrenamiento 
    def train_raw(): #Funcion que entrena el modelo
        #Aplicamos una regresion lineal y le decimos que ajuste los datos para raw y y
        linear_model.LinearRegression().fit(raw, y) 
    
    #Debemos verificar el entramiento del modelo con los datos escalados, esta no es la manera mas optima
    def train_scaled():
        linear_model.LinearRegression().fit(scaled, y)
    
    def train_z_scaled():
        linear_model.LinearRegression().fit(zscore, y)
    
    #Evaluamos el tiempo que tarda en ejecutar el modelo
    raw_time = timeit.timeit(train_raw, number=100) #Number son las veces que repite el codigo para calcular
    scaled_time = timeit.timeit(train_scaled, number=100)
    z_scaled_time = timeit.timeit(train_z_scaled, number=100)
    print("train raw : {}".format(raw_time))
    print("train scaled : {}".format(scaled_time))
    print("train z_scaled : {}".format(z_scaled_time))

    df = pd.read_csv("cars.csv")
    df.price_usd.hist()
    plt.show()

    #Transformacion con la tangente hipervolica tanh(x)
    p = 10000 #Numero que nos ayuda a ajustar la curva sin ella esta transformacion nos lo deja en 1
    df.price_usd.apply(lambda x: np.tanh(x/p)).hist()
    plt.show()

if __name__ == "__main__":
    run()