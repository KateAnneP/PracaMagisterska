import pandas as pd
import functions as f

def main():
    # Dane
    dane = pd.read_csv('dane/adepression.csv', delimiter=';')
    nazwa_tabeli = "depression"
    attributes_info = [
        ("f_0_d", "symbolic"),
        ("f_1_d", "symbolic"),
        ("f_2_d", "symbolic"),
        ("f_3_d", "symbolic"),
        ("f_4_d", "symbolic"),
        ("f_5_d", "symbolic"),
        ("f_6_d", "symbolic"),
        ("f_7_d", "symbolic"),
        ("f_8_d", "symbolic"),
        ("f_9_d", "symbolic"),
        ("group", "symbolic"),
    ]

    duplicates = dane.duplicated().sum()
    # print(f"Duplikaty: {duplicates}")

    do_usuniecia = [34, 68]
    dane = dane.drop(dane.index[do_usuniecia])
    dane = dane.reset_index(drop=True)

    # Przygotowanie atrybutów:
    dane_split = dane['attributes'].str.split('|', expand=True)
    dane_split.columns = [f'f_{i}' for i in range(dane_split.shape[1])]
    dane = pd.concat([dane_split, dane['date'], dane['email'], dane['gender']], axis=1)
    features = dane.iloc[:, 0:10]  # cechy do grupowania, bez innych kolumn w tabeli
    mapping = {'0': '3', '1': '2', '2': '1', '3': '0'}
    cols = [1, 2, 3, 6, 7]
    for col in cols:
        features.iloc[:, col] = features.iloc[:, col].map(mapping)

    #################################################
    # Nr grupowania: 1-Kmeans, 2-hierarchiczne, 3-DBSCAN
    #f.grupowanie_robocze(features, nazwa_tabeli, attributes_info, 1)
    #f.grupowanie_robocze(features, nazwa_tabeli, attributes_info, 2)

    reguly_test, reguly_caly = f.grupowanie(features, nazwa_tabeli, attributes_info, 1)
    f.stabilnosc(reguly_test, reguly_caly)

    reguly_test, reguly_caly = f.grupowanie(features, nazwa_tabeli, attributes_info, 2)
    f.stabilnosc(reguly_test, reguly_caly)

    grupyDBSCAN = f.grupowanie(features, nazwa_tabeli, attributes_info, 3)

if __name__ == "__main__":
    main()