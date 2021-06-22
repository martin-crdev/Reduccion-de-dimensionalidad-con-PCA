import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def run():
    iris = sns.load_dataset("iris")
    iris.head()

    #scatterplot
    sns.scatterplot(data=iris, x = "sepal_length", y = "petal_length", hue = "species")
    plt.show()

    #joinplot
    sns.jointplot(data=iris, x = "sepal_length", y = "petal_length", hue = "species")
    plt.show()

    #box plot
    sns.boxplot(x = "species", y = "sepal_length", data = iris)
    plt.show()

    #bar plot
    sns.barplot(x = "species", y = "sepal_length", data = iris)
    plt.show()

if __name__ == "__main__":
    run()