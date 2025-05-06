import pandas as pd
from pandas.api.types import CategoricalDtype
from datetime import datetime
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
    dane2 = dane.drop(dane.index[do_usuniecia])
    dane2 = dane.reset_index(drop=True)

    # Przygotowanie atrybutów:
    dane_split = dane['attributes'].str.split('|', expand=True)
    dane_split.columns = [f'f_{i}' for i in range(dane_split.shape[1])]
    dane = pd.concat([dane_split, dane['date'], dane['email'], dane['gender']], axis=1)
    features = dane.iloc[:, 0:10]  # cechy do grupowania, bez innych kolumn w tabeli

    dane2_split = dane2['attributes'].str.split('|', expand=True)
    dane2_split.columns = [f'f_{i}' for i in range(dane2_split.shape[1])]
    dane2 = pd.concat([dane2_split, dane2['date'], dane2['email'], dane2['gender']], axis=1)
    features2 = dane2.iloc[:, 0:10]  # cechy do grupowania, bez innych kolumn w tabeli

    featuresA = features.rename(columns=lambda x: f"{x}_a")  # Zmiana nazwy, żeby można było rozróżnić cechy
    featuresD = features2.rename(columns=lambda x: f"{x}_d")

    all_features = pd.concat([featuresA, featuresD], axis=1)
    indexes = []

    # Resetowanie indeksów
    for i in range(len(all_features)):
        if all_features.iloc[i].isna().any():
            indexes.append(i)

    all_features = all_features.drop(all_features.index[indexes])
    all_features = all_features.reset_index(drop=True)

    # Zbiór danych users
    users = pd.read_csv('dane/ausers.csv', delimiter=';')

    current_year = datetime.now().year  # Bieżący rok
    single_ad = users.copy()

    # ------------------------------------------------------------------------------
    do_usuniecia = []
    single_ad['partner_education'] = None
    single_ad['partner_age'] = None

    for i in range(len(single_ad)):
        # if single_ad['gender'].iloc[i] == "M":
        id = single_ad['email']
        partner = single_ad['partner_email'].iloc[i]
        znaleziono_partnera = False
        for j in range(len(single_ad)):
            if single_ad['email'].iloc[j] == partner:
                znalezniono_partnera = True
                partner_wiek = current_year - single_ad['birth_year'].iloc[j]
                partner_edukacja = single_ad['education'].iloc[j]
        if znaleziono_partnera == False:
            single_ad.loc[i, 'partner_education'] = partner_edukacja
            single_ad.loc[i, 'partner_age'] = partner_wiek
        else:
            do_usuniecia.append(i)

    single_ad = single_ad.drop(single_ad.index[do_usuniecia]).reset_index(
        drop=True)  # Jak nie znaleziono partnera, to nie ma sensu klasyfikować tej osoby w tym zestawieniu
    # ------------------------------------------------------------------------------

    # Eliminowanie NaN
    for i in range(len(single_ad)):
        if pd.isna(single_ad['doughter_birth_years'].iloc[i]):
            single_ad.loc[i, 'doughter_birth_years'] = "0"
        if pd.isna(single_ad['son_birth_years'].iloc[i]):
            single_ad.loc[i, 'son_birth_years'] = "0"

    # Ilość synów i córek
    single_ad['no_of_daughters'] = single_ad['doughter_birth_years'].apply(
        lambda x: 0 if x == '0' else len(x.split(','))
    )
    single_ad['no_of_sons'] = single_ad['son_birth_years'].apply(
        lambda x: 0 if x == '0' else len(x.split(','))
    )
    # Wiek najmłodszego dziecka
    single_ad['child_birth_years'] = single_ad['doughter_birth_years'].fillna('') + ',' + single_ad[
        'son_birth_years'].fillna('')
    single_ad['child_birth_years'] = single_ad['child_birth_years'].str.strip(',')
    single_ad['age_oldest_child'] = current_year - single_ad['child_birth_years'].apply(
        lambda x: max(map(int, x.split(','))))
    single_ad['age_oldest_child'] = single_ad['age_oldest_child'].apply(
        lambda x: 0 if x == current_year else x
    )

    # Średnia wieku dzieci
    def calculate_average_years(date_string):
        years = date_string.split(',')  # Daty rozdzielone przecinkami
        age_differences = [current_year - int(year.strip()) for year in years if year.strip().isdigit() and year != '0']
        return round(sum(age_differences) / len(age_differences)) if age_differences else 0

    single_ad['average_child_age'] = single_ad['child_birth_years'].apply(calculate_average_years).apply(
        lambda x: 0 if x == current_year else x
    )

    # Edukacja
    education_order = CategoricalDtype(categories=['średnie', 'wyższe'], ordered=True)  # 0 - średnie, 1 - wyższe
    single_ad['education'] = single_ad['education'].astype(education_order)
    single_ad['education'] = single_ad['education'].cat.codes
    single_ad['partner_education'] = single_ad['partner_education'].astype(education_order)
    single_ad['partner_education'] = single_ad['partner_education'].cat.codes

    # Mapowanie płci na liczby
    gender_mapping = {'K': 0, 'M': 1}
    single_ad['gender'] = single_ad['gender'].map(gender_mapping)

    # Kolumna rok urodzenia - zamiana na wiek
    single_ad['age'] = current_year - single_ad['birth_year']  # Tworzenie nowej kolumny z wiekiem
    single_ad['years_of_mariage'] = current_year - single_ad['mariage_year']  # Kolumna ze stażem małżeństwa
    single_ad['years_of_relationship'] = current_year - single_ad['relation_start_year']  # Kolumna ze stażem związku

    # Usuwanie niepotrzebnych do grupowania kolumn
    single_ad = single_ad.drop(
        columns=['birth_year', 'email', 'partner_email', 'doughter_birth_years', 'son_birth_years', 'child_birth_years',
                 'mariage_year', 'relation_start_year'])

    single_ad = pd.concat([all_features, single_ad],
                          axis=1)  # data frame z danymi dot. anxiety i depression dla pojedynczych użytkowników
    # print(single_ad)

    indexes = []

    for i in range(len(single_ad)):
        if single_ad.iloc[i].isna().any():
            indexes.append(i)

    single_ad = single_ad.drop(single_ad.index[indexes])
    features_ad = single_ad.reset_index(drop=True)
    print(list(features_ad))

    #################################################
    # Nr grupowania: 1-Kmeans, 2-hierarchiczne, 3-DBSCAN
    grupy_Kmeans = f.grupowanie(features_ad, nazwa_tabeli, attributes_info, 1)
    grupy_hierarchiczne = f.grupowanie(features_ad, nazwa_tabeli, attributes_info, 2)
    #grupyDBSCAN = f.grupowanie(features_ad, nazwa_tabeli, attributes_info, 3)
    #f.drzewoDecyzyjne(grupy_Kmeans)

if __name__ == "__main__":
    main()