import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def run():
    iris = sns.load_dataset("iris")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(
        iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values
    )

    covariance_matrix = np.cov(scaled.T)

    #Diagrama de dispersion de todo el dataset
    sns.pairplot(iris)

    #Diagrama de dispersion solo de los atributos petal_length y petal_width
    sns.jointplot(x = iris["petal_length"], y = iris["petal_width"], data=iris)

    #El mismo diagrama pero con los datos estandarizados 
    sns.jointplot(x = scaled[:, 2], y = scaled[:, 3], data=iris)

    #Funcion eigen para calcular la descomposicion en vectores y valores propio
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

    #Cada vector propio es una direccion principal a lo largo de la cual capturamos varianza de los datos originales
    variance_explained = [] #Varianza se puede calcular proporcional a la relacion entre el valor propio particular
    for i in eigen_values:  #y la suma de todos los valores propios dentro de la matriz
        variance_explained.append((i / sum(eigen_values))*100)
    print(variance_explained)

    #PCA con scikit
    pca = PCA(n_components=2) #Reduce en 2 componenetes
    pca.fit(scaled)

    pca.explained_variance_ratio_ #Ver el radio de la varianza 

    #Nuevo conjunto de variables que reciben los datos originales transformados por la resultante
    #de la descomposicion de valores y vectores propios
    reduced_scaled = pca.transform(scaled)

    #Asignamos las componentes reducidas como nuevas columnas al dataframe 
    iris["pca_1"] = reduced_scaled[:, 0]
    iris["pca_2"] = reduced_scaled[:, 1]

    #visualizacion de conjunto de datos reducido en esas componentes principales en dos dimensiones
    sns.jointplot(iris["pca_1"], iris["pca_2"], hue = iris["species"], data=iris)
    plt.show()


if __name__ == "__main__":
    run()