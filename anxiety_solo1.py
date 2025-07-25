import pandas as pd
import functions as f

def main():
    # Dane
    dane = pd.read_csv('excluded/dane/aanxiety.csv', delimiter=';')
    nazwa_tabeli = "anxiety"
    attributes_info = [
        ("f_0_a", "symbolic"),
        ("f_1_a", "symbolic"),
        ("f_2_a", "symbolic"),
        ("f_3_a", "symbolic"),
        ("f_4_a", "symbolic"),
        ("f_5_a", "symbolic"),
        ("f_6_a", "symbolic"),
        ("f_7_a", "symbolic"),
        ("f_8_a", "symbolic"),
        ("f_9_a", "symbolic"),
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
    mapping = {'0': '3', '1': '2', '2': '1', '3': '0'}
    features.iloc[:, 3] = features.iloc[:, 3].map(mapping)

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