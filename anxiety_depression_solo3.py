import pandas as pd
import functions as f

def main():
    # Dane
    dane = pd.read_csv('dane/aanxiety.csv', delimiter=';')
    dane2 = pd.read_csv('dane/adepression.csv', delimiter=';')

    nazwa_tabeli = "anxiety_depression"
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

    do_usuniecia = [39, 68, 91]
    dane = dane.drop(dane.index[do_usuniecia])
    dane = dane.reset_index(drop=True)

    do_usuniecia2 = [34, 68]
    dane2 = dane2.drop(dane2.index[do_usuniecia2])
    dane2 = dane2.reset_index(drop=True)

    # Przygotowanie atrybutów:
    dane_split = dane['attributes'].str.split('|', expand=True)
    dane_split.columns = [f'f_{i}_a' for i in range(dane_split.shape[1])]
    dane = pd.concat([dane_split, dane['email']], axis=1)

    dane2_split = dane2['attributes'].str.split('|', expand=True)
    dane2_split.columns = [f'f_{i}_d' for i in range(dane2_split.shape[1])]
    dane2 = pd.concat([dane2_split, dane2['email']], axis=1)

    all_features = (pd.merge(dane, dane2, on='email', how='inner')).drop(columns=['email'])
    indexes = []

    #Resetowanie indeksów
    for i in range(len(all_features)):
        if all_features.iloc[i].isna().any():
            indexes.append(i)

    all_features = all_features.drop(all_features.index[indexes])
    all_features = all_features.reset_index(drop=True)

    #################################################
    # Nr grupowania: 1-Kmeans, 2-hierarchiczne, 3-DBSCAN
    reguly_test, reguly_caly = f.grupowanie(all_features, nazwa_tabeli, attributes_info, 1)  # clusters też wcześniej dawało
    #f.stabilnosc(reguly_test, reguly_caly)
    #reguly_test, reguly_caly = f.grupowanie(all_features, nazwa_tabeli, attributes_info, 2)
    #f.stabilnosc(reguly_test, reguly_caly)
    #grupyDBSCAN = f.grupowanie(all_features, nazwa_tabeli, attributes_info, 3)
    #f.drzewoDecyzyjne(grupy_Kmeans)

if __name__ == "__main__":
    main()