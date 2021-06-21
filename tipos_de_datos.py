import pandas as pd

df = pd.read_csv("cars.csv")

def run():
    print(df.dtypes)

if __name__ == "__main__":
    run()