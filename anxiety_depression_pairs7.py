import pandas as pd
import functions as f
import users_dane_prepared

def main():
    # Dane
    dane = pd.read_csv('excluded/dane/aanxiety.csv', delimiter=';')
    dane2 = pd.read_csv('excluded/dane/adepression.csv', delimiter=';')

    nazwa_tabeli = "anxiety_depression_pairs"
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
        ("child_number", "symbolic"),
        ("education", "symbolic"),
        ("gender", "symbolic"),
        ("partner_education", "symbolic"), #23
        ("partner_age", "symbolic"),
        ("no_of_daughters", "symbolic"),
        ("no_of_sons", "symbolic"),
        ("age_oldest_child", "symbolic"),
        ("average_child_age", "symbolic"),
        ("age", "symbolic"),
        ("years_of_mariage", "symbolic"),
        ("years_of_relationship", "symbolic"),
        ("f_0_a_partner", "symbolic"), #32
        ("f_1_a_partner", "symbolic"),
        ("f_2_a_partner", "symbolic"),
        ("f_3_a_partner", "symbolic"), #35
        ("f_4_a_partner", "symbolic"),
        ("f_5_a_partner", "symbolic"),
        ("f_6_a_partner", "symbolic"),
        ("f_7_a_partner", "symbolic"),
        ("f_8_a_partner", "symbolic"), #40
        ("f_9_a_partner", "symbolic"),
        ("f_0_d_partner", "symbolic"),
        ("f_1_d_partner", "symbolic"),
        ("f_2_d_partner", "symbolic"),
        ("f_3_d_partner", "symbolic"),
        ("f_4_d_partner", "symbolic"),
        ("f_5_d_partner", "symbolic"),
        ("f_6_d_partner", "symbolic"), #48
        ("f_7_d_partner", "symbolic"),
        ("f_8_d_partner", "symbolic"),
        ("f_9_d_partner", "symbolic"),
        ("group", "symbolic")
    ]

    duplicates = dane.duplicated().sum()
    # print(f"Duplikaty: {duplicates}")

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
    mapping = {'0': '3', '1': '2', '2': '1', '3': '0'}
    dane.iloc[:, 3] = dane.iloc[:, 3].map(mapping)

    dane2_split = dane2['attributes'].str.split('|', expand=True)
    dane2_split.columns = [f'f_{i}_d' for i in range(dane2_split.shape[1])]
    dane2 = pd.concat([dane2_split, dane2['email']], axis=1)
    cols = [1, 2, 3, 6, 7]
    for col in cols:
        dane2.iloc[:, col] = dane2.iloc[:, col].map(mapping)

    all_features = (pd.merge(dane, dane2, on='email', how='inner'))
    users = users_dane_prepared.prepared
    user_ad = (pd.merge(all_features, users, on='email', how='inner'))

    z_partnerami = []
    for index, user in user_ad.iterrows():
        email = user['partner_email']
        cechy = user_ad[user_ad['partner_email'] == email].iloc[:, 0:21] #cechy dla partnera
        cechy = cechy.iloc[0].rename(lambda x: f"{x}_partner")  # Zmiana nazwy, żeby można było rozróżnić cechy
        nowy_user = pd.concat([user, cechy])
        z_partnerami.append(nowy_user)

    user_ad = pd.DataFrame(z_partnerami)
    user_ad = user_ad.drop(columns=['email', 'partner_email','email_partner'])
    indexes = []

    # Resetowanie indeksów
    for i in range(len(user_ad)):
        if user_ad.iloc[i].isna().any():
            indexes.append(i)

    user_ad = user_ad.drop(user_ad.index[indexes])
    user_ad = user_ad.reset_index(drop=True)

    print(max(user_ad['years_of_mariage']))
    print(min(user_ad['years_of_mariage']))
    print(max(user_ad['years_of_relationship']))
    print(min(user_ad['years_of_relationship']))
    print(max(user_ad['age_oldest_child']))
    print(min(user_ad['age_oldest_child']))
    print(max(user_ad['average_child_age']))
    print(min(user_ad['average_child_age']))


    #################################################
    # Nr grupowania: 1-Kmeans, 2-hierarchiczne, 3-DBSCAN
    #f.grupowanie_robocze(user_ad, nazwa_tabeli, attributes_info, 1)
    #f.grupowanie_robocze(user_ad, nazwa_tabeli, attributes_info, 2)

    reguly_test, reguly_caly = f.grupowanie(user_ad, nazwa_tabeli, attributes_info, 1)
    #f.stabilnosc(reguly_test, reguly_caly)

    reguly_test, reguly_caly = f.grupowanie(user_ad, nazwa_tabeli, attributes_info, 2)
    f.stabilnosc(reguly_test, reguly_caly)

    grupyDBSCAN = f.grupowanie(user_ad, nazwa_tabeli, attributes_info, 3)

if __name__ == "__main__":
    main()