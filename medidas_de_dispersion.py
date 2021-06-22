import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run():
    df = pd.read_csv("cars.csv")

    #Desviacion estandar
    df["price_usd"].std()

    #Rango = valor mas - valor min
    rango = df["price_usd"].max() - df["price_usd"].min()

    #Quartiles
    #El metodo quantile es una manera de subdividir datos de una manera homogenea
    median = df["price_usd"].median() #Declaramos una variable con la media
    Q1 = df["price_usd"].quantile(q=0.25) #Declaramos el quartil 1 que abarca 1/4 de los datos
    Q3 = df["price_usd"].quantile(q=0.75) #Declaramos el quartil 3 que abarca 3/4 de los datos
    min_val = df["price_usd"].quantile(q=0) #q=0 indica el valor minimo
    max_val = df["price_usd"].quantile(q=1) #q=1 indica el valor maximo 

    #Rango interquartil
    iqr = Q3 - Q1 

    #Limites para deteccion de outliers (datos simetricamete distribuidos)
    minlimit = Q1 - 1.5*iqr
    maxlimit = Q3 + 1.5*iqr

    #Diagrama de caja con la columna de precios
    sns.boxplot(x = "price_usd", data= df)
    plt.show()

    #Los diagramas de caja nos permiten trabajar mejor cpm variables categoricas
    # porque nos dejan visualizar mejor la distribucion de los datos 
    sns.boxenplot(x = "engine_fuel", y = "price_usd", data= df)
    plt.show()



if __name__ == "__main__":
    run()