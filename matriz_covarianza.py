import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def run():
    iris = sns.load_dataset("iris")

    #Mostrar correlaciones con parejas de datos
    sns.pairplot(iris, hue = "species")
    plt.show()

    #Escalamiento de los datos 
    scaler = StandardScaler()
    
    #La data escalada agarra el escalador y aplicamos a nuestros datos
    scaled = scaler.fit_transform(
        iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    ) 

    #Matriz de covarianza
    covariance_matrix = np.cov(scaled.T) #.T es para indicarle que use la matriz traspuesta de los datos

    #Mapa de calor 
    plt.figure(figsize=(10,10))
    sns.set(font_scale=1.5)
    hm = sns.heatmap(covariance_matrix, #Indicamos la matriz a visualizar
                     cbar=True, #Muestre la escala de colores 
                     annot=True, #Muestre los valores dentro de la matriz
                     square=True, #Que se muestre la matriz cuadrada
                     fmt='.2f', #Parametros para el tipo y tamaño de letra
                     annot_kws={'size': 12},
                     yticklabels=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
                     xticklabels=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    plt.show()


if __name__ == "__main__":
    run()