import pandas as pd
import matplotlib.pyplot as plt #Libreria para graficar
import seaborn as sns #Libreria para la visualizacion y distribucion de datos


df = pd.read_csv("cars.csv")

def run():
    df["price_usd"].mean() #Promedio de la columna price_usd del archivo cars.csv
    df["price_usd"].median() #Mediana

    #Plot metodo para graficar, .hist en forma de histograma, bins para indicar los intervalos 
    df["price_usd"].plot.hist(bins=20)
    plt.show() #Metodo visualizar la grafica

    #Usamos la libreria seaborn para hacer un histograma del precio por cada marca
    #No se recomientda graficar gran cantidad de datos que dificulte la visualizacion
    sns.displot(df, x = "price_usd", hue = "manufacturer_name")
    plt.show()

    #Hacemos un histograma del precio por cada tipo de motor pero los muestra sobrepuestos
    #Lo que lo hace dificil de visualizar
    sns.displot(df, x = "price_usd", hue = "engine_type")
    plt.show()

    #Agregamos el parametro multiple=stack para que nos lo muestre mas claro
    #Pero si hay una desporcion en los datos no sera visible todos los datos
    sns.displot(df, x = "price_usd", hue = "engine_type", multiple = "stack")
    plt.show()

    #Se agrupa los datos por el tipo de motor que tienen y hace un conteo 
    df.groupby("engine_type").count()

    #Filtramos el data frame con dos condiciones que la marca sea Audi y el modelo Q7
    Q7_df = df[(df["manufacturer_name"] == "Audi") & (df["model_name"] == "Q7")]
    #seaborn tiene su propio metodo para crear histogramas, le pasamos el data frame filtrado Q7_df
    #donde lo filtre por el precio por año del auto
    sns.histplot(Q7_df, x = "price_usd", hue = "year_produced")
    plt.show()


if __name__ == "__main__":
    run()