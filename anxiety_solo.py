import pandas as pd
import functions as f

def main():
    # Dane
    dane = pd.read_csv('dane/aanxiety.csv', delimiter=';')
    nazwa_tabeli = "anxiety"
    attributes_info = [
        ("f_0", "symbolic"),
        ("f_1", "symbolic"),
        ("f_2", "symbolic"),
        ("f_3", "symbolic"),
        ("f_4", "symbolic"),
        ("f_5", "symbolic"),
        ("f_6", "symbolic"),
        ("f_7", "symbolic"),
        ("f_8", "symbolic"),
        ("f_9", "symbolic"),
        ("group", "symbolic"),
    ]

    duplicates = dane.duplicated().sum()
    # print(f"Duplikaty: {duplicates}")

    do_usuniecia = [39, 68, 91]
    dane = dane.drop(dane.index[do_usuniecia])
    dane = dane.reset_index(drop=True)

    # Przygotowanie atrybutów:
    dane_split = dane['attributes'].str.split('|', expand=True)
    dane_split.columns = [f'f_{i}' for i in range(dane_split.shape[1])]
    dane = pd.concat([dane_split, dane['date'], dane['email'], dane['gender']], axis=1)
    features = dane.iloc[:, 0:10]  # cechy do grupowania, bez innych kolumn w tabeli

    #################################################

    #f.grupowanie1_Kmeans(features, nazwa_tabeli, attributes_info)
    f.grupowanie2_hierarchiczne(features, nazwa_tabeli, attributes_info)
    #f.grupowanie3_DBSCAN(features, nazwa_tabeli, attributes_info)

if __name__ == "__main__":
    main()