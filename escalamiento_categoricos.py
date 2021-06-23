import pandas as pd
import sklearn.preprocessing as prepocessing #Libreria para el procesamiento de datos categoricos y numericos

def run():
    df = pd.read_csv("cars.csv")

    #Metodo para obtener los datos categoricos en numericos en one-hot de pandas
    print(pd.get_dummies(df["engine_type"]))

    #Metodo de codificacion de variables categoricas con sklearn
    encoder = prepocessing.OneHotEncoder(handle_unknown="ignore") #"ignore permite procesar datos que no venian en el dataset"
    
    #ajustarlo a las categorias de nuestro dataset
    encoder.fit(df[["engine_type"]].values)
    #Especificamos que haga la codificacion en los atributos gasoline, diesel y aceite
    print(encoder.transform([["gasoline"], ["diesel"], ["aceite"]]).toarray())

    #Variables numericas discretas pueden ser codificadas como categoricas
    encoder.fit(df[["year_produced"]].values)
    print(encoder.transform([[2016], [2009], [190]]).toarray())

if __name__ == "__main__":
    run()