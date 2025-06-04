from datetime import datetime
import pandas as pd
from pandas import CategoricalDtype
import users_dane_prepared as dane_user
import functions as f

def main():
    # Dane
    dane = pd.read_csv('dane/aanxiety.csv', delimiter=';')
    nazwa_tabeli = "anxiety_user"
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
        ("child_number", "symbolic"),
        ("education", "symbolic"),
        ("gender", "symbolic"),
        ("partner_education", "symbolic"),
        ("partner_age", "symbolic"),
        ("no_of_daughters", "symbolic"),
        ("no_of_sons", "symbolic"),
        ("age_oldest_child", "symbolic"),
        ("average_child_age", "symbolic"),
        ("age", "symbolic"),
        ("years_of_mariage", "symbolic"),
        ("years_of_relationship", "symbolic"),
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

    dane_prepared = dane.drop(columns=['date','gender'])
    single_a = pd.merge(dane_prepared, dane_user.prepared, on='email', how='inner').drop(columns=['email','partner_email'])
      # data frame z danymi dot. anxiety dla pojedynczych użytkowników

    indexes = []

    for i in range(len(single_a)):
        if single_a.iloc[i].isna().any():
            indexes.append(i)

    single_a = single_a.drop(single_a.index[indexes])
    features_a = single_a.reset_index(drop=True)
    print(list(features_a))

    #################################################
    # Nr grupowania: 1-Kmeans, 2-hierarchiczne, 3-DBSCAN
    reguly_test, reguly_caly = f.grupowanie(features_a, nazwa_tabeli, attributes_info, 1)  # clusters też wcześniej dawało
    #f.stabilnosc(reguly_test, reguly_caly)
    #reguly_test, reguly_caly = f.grupowanie(features_a, nazwa_tabeli, attributes_info, 2)
    #f.stabilnosc(reguly_test, reguly_caly)
    #grupyDBSCAN = f.grupowanie(features_a, nazwa_tabeli, attributes_info, 3)
    #f.drzewoDecyzyjne(grupy_Kmeans)

if __name__ == "__main__":
    main()