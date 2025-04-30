import functions as f
import pandas as pd

#########################################
#Dane

dane = pd.read_csv('dane/aanxiety.csv', delimiter=';')
nazwa_tabeli = "anxiety"
attributes_info = [
    ("f_0","symbolic"),
    ("f_1","symbolic"),
    ("f_2","symbolic"),
    ("f_3","symbolic"),
    ("f_4","symbolic"),
    ("f_5","symbolic"),
    ("f_6","symbolic"),
    ("f_7","symbolic"),
    ("f_8","symbolic"),
    ("f_9","symbolic"),
    ("group","symbolic"),
]

duplicates = dane.duplicated().sum()
#print(f"Duplikaty: {duplicates}")

do_usuniecia = [39, 68, 91]
dane = dane.drop(dane.index[do_usuniecia])
dane = dane.reset_index(drop=True)

# Przygotowanie atrybutów:
dane_split = dane['attributes'].str.split('|', expand=True)
dane_split.columns = [f'f_{i}' for i in range(dane_split.shape[1])]
dane = pd.concat([dane_split, dane['date'], dane['email'], dane['gender']], axis=1)
features = dane.iloc[:,0:10] #cechy do grupowania, bez innych kolumn w tabeli

#################################################
# -- Grupowanie K-means ---
def grupowanie1_Kmeans(features, nazwa_tabeli, attributes_info):

    clusters_Kmeans = []
    centroids_Kmeans = []
    inertia_Kmeans = []

    #Tablice używane do wykresów
    silhouettes_Kmeans = []
    dbi_Kmeans = []
    calinski_harabasz_Kmeans = []

    #Słowniki używane do właściwej ilości grup
    best_silhouette_Kmeans = {0: -1}
    best_dbi_Kmeans = {0: -1}
    best_ch_Kmeans = {0: -1}
    best_indexes_Kmeans = []

    for i in range(1,10):
        clusters, centroids, inertia = f.grupowanieKmeans(i, features)
        clusters_Kmeans.append(clusters)
        centroids_Kmeans.append(centroids)
        inertia_Kmeans.append(inertia)
        if(i > 1):

            #Wyszukiwanie najlepszej wartości miary Silhouette
            current_silhouette = f.miaraSilhouette(features, clusters)
            best_silhouette = list(best_silhouette_Kmeans.values())[0]
            if current_silhouette >= best_silhouette:
                best_silhouette_Kmeans.clear()
                best_silhouette_Kmeans[i] = current_silhouette

            #Wyszukiwanie najlepszej wartości miary DBI
            current_dbi = f.miaraDBI(features, clusters)
            best_dbi = list(best_dbi_Kmeans.values())[0]
            if current_silhouette >= best_dbi:
                best_dbi_Kmeans.clear()
                best_dbi_Kmeans[i] = current_dbi

            # Wyszukiwanie najlepszej wartości miary DBI
            current_ch = f.miaraCalinskiHarabasz(features, clusters)
            best_ch = list(best_ch_Kmeans.values())[0]
            if current_ch >= best_ch:
                best_ch_Kmeans.clear()
                best_ch_Kmeans[i] = current_ch

            silhouettes_Kmeans.append(current_silhouette)
            dbi_Kmeans.append(current_dbi)
            calinski_harabasz_Kmeans.append(current_ch)

    # Wykresy miar jakości grupowania
    # f.pokazWykresLokcia(inertia_Kmeans, 10)
    # f.pokazWykresSilhouette(silhouettes_Kmeans, 10)
    # f.pokazWykresDBI(dbi_Kmeans, 10)
    # f.pokazWykresCalinskiHarabasz(calinski_harabasz_Kmeans, 10)

    best_indexes_Kmeans = [list(best_silhouette_Kmeans.keys())[0], list(best_dbi_Kmeans.keys())[0], list(best_ch_Kmeans.keys())[0]]
    print(f"Na ile grup powinny być podzielone dane w grupowaniu K-means wg miar jakości grupowania: {best_indexes_Kmeans}")

    #Dla każdej liczby grup, która jest najlepsza wg miar jakości wykonywane jest przypisanie grup do danych, a potem stworzenie reguł decyzyjnych i eksport do RSES
    for i in dict.fromkeys(best_indexes_Kmeans):
        clustersG = clusters_Kmeans[i]
        grupyKmeans = f.przypisanieGrup(features, clustersG)
        reguly = f.regulyDecyzyjne(grupyKmeans)
        f.eksportDoRSES(attributes_info, reguly, nazwa_tabeli, f"Kmeans_{nazwa_tabeli}_grupy{i}.tab")

#################################################
# --- Grupowanie hierarchiczne ---

def grupowanie2_hierarchiczne(features, nazwa_tabeli, attributes_info):

    clusters_hierarchiczne = []
    silhouettes_hierarchiczne = []
    dbi_hierarchiczne = []
    calinski_harabasz_hierarchiczne = []
    for i in range(1,10):
        clusters = f.grupowanieHierarchiczne(i, features, "complete", "euclidean")
        clusters_hierarchiczne.append(clusters)
        if(i > 1):
            silhouettes_hierarchiczne.append(f.miaraSilhouette(features, clusters))
            dbi_hierarchiczne.append(f.miaraDBI(features, clusters))
            calinski_harabasz_hierarchiczne.append(f.miaraCalinskiHarabasz(features, clusters))

    f.pokazWykresSilhouette(silhouettes_hierarchiczne, 10)
    f.pokazWykresDBI(dbi_hierarchiczne, 10)
    f.pokazWykresCalinskiHarabasz(calinski_harabasz_hierarchiczne, 10)

#################################################
# --- Grupowanie DBSCAN ---
# min_samples = features.shape[1] * 2
# f.wykresNajblizszychOdleglosci(features, min_samples)
# eps = 5.0
# clustersDBSCAN = f.grupowanieDBSCAN(features, eps, min_samples)

################################################
#Teraz przypisywanie do danych
#
# grupyHierarchiczne = f.przypisanieGrup(features, clusters_hierarchiczne)
# grupyDBSCAN = f.przypisanieGrup(features, clustersDBSCAN)




#Jeszcze zrób to porównanie miar jakości

#Dla wszystkich najlepszych, które wyjdą. Jeśli dla sil wyjdzie 2,
# dla dbi 3, a dla calinski 4, to dla wszystkich spisujemy wyniki

#Zrób do tabeli zestawienie wyników wszystkich miar jakości dla wszystkich grupowań

#Dopisz o doborze eps i min_samples do teorii, ale nie masz źródła

#Główny algorytm tylko jako pseudokod, a resztę jako algorytmy???