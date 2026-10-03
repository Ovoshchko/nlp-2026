import pandas as pd
import matplotlib.pyplot as plt

def hist_count_by_column(df: pd.DataFrame, column_name: str, bins: int = 20):
    counts, edges, patches = plt.hist(
        df[column_name], bins=20, edgecolor="black"
    )
    plt.xticks(edges, rotation=45)
    plt.grid(True)
    plt.xlabel(f"Кол-во {column_name}")
    plt.ylabel("Частота")
    plt.title("Распределение значений в Train")
    plt.show()