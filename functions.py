import utils as f

#Zrób do tabeli zestawienie wyników wszystkich miar jakości dla wszystkich grupowań
#Dopisz o doborze eps i min_samples do teorii, ale nie masz źródła
#Główny algorytm tylko jako pseudokod, a resztę jako algorytmy???

#########################################

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

    for i in range(2,10):
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
    f.pokazWykresLokcia(inertia_Kmeans, 10)
    f.pokazWykresSilhouette(silhouettes_Kmeans, 10)
    f.pokazWykresDBI(dbi_Kmeans, 10)
    f.pokazWykresCalinskiHarabasz(calinski_harabasz_Kmeans, 10)

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

    # Tablice używane do wykresów
    silhouettes_hierarchiczne = []
    dbi_hierarchiczne = []
    calinski_harabasz_hierarchiczne = []

    # Słowniki używane do właściwej ilości grup
    best_silhouette_hierarchiczne = {0: -1}
    best_dbi_hierarchiczne = {0: -1}
    best_ch_hierarchiczne = {0: -1}
    best_indexes_hierarchiczne = []

    for i in range(2,10):
        clusters = f.grupowanieHierarchiczne(i, features, "complete", "euclidean")
        clusters_hierarchiczne.append(clusters)

        # Wyszukiwanie najlepszej wartości miary Silhouette
        current_silhouette = f.miaraSilhouette(features, clusters)
        best_silhouette = list(best_silhouette_hierarchiczne.values())[0]
        if current_silhouette >= best_silhouette:
            best_silhouette_hierarchiczne.clear()
            best_silhouette_hierarchiczne[i] = current_silhouette

        # Wyszukiwanie najlepszej wartości miary DBI
        current_dbi = f.miaraDBI(features, clusters)
        best_dbi = list(best_dbi_hierarchiczne.values())[0]
        if current_silhouette >= best_dbi:
            best_dbi_hierarchiczne.clear()
            best_dbi_hierarchiczne[i] = current_dbi

        # Wyszukiwanie najlepszej wartości miary DBI
        current_ch = f.miaraCalinskiHarabasz(features, clusters)
        best_ch = list(best_ch_hierarchiczne.values())[0]
        if current_ch >= best_ch:
            best_ch_hierarchiczne.clear()
            best_ch_hierarchiczne[i] = current_ch

        silhouettes_hierarchiczne.append(current_silhouette)
        dbi_hierarchiczne.append(current_dbi)
        calinski_harabasz_hierarchiczne.append(current_ch)

    #Wykresy miar jakości grupowania
    f.pokazWykresSilhouette(silhouettes_hierarchiczne, 10)
    f.pokazWykresDBI(dbi_hierarchiczne, 10)
    f.pokazWykresCalinskiHarabasz(calinski_harabasz_hierarchiczne, 10)

    best_indexes_hierarchiczne = [list(best_silhouette_hierarchiczne.keys())[0], list(best_dbi_hierarchiczne.keys())[0], list(best_ch_hierarchiczne.keys())[0]]
    print(f"Na ile grup powinny być podzielone dane w grupowaniu hierarchicznym wg miar jakości grupowania: {best_indexes_hierarchiczne}")

    # Dla każdej liczby grup, która jest najlepsza wg miar jakości wykonywane jest przypisanie grup do danych, a potem stworzenie reguł decyzyjnych i eksport do RSES
    for i in dict.fromkeys(best_indexes_hierarchiczne):
        clustersG = clusters_hierarchiczne[i]
        grupyHierarchiczne = f.przypisanieGrup(features, clustersG)
        reguly = f.regulyDecyzyjne(grupyHierarchiczne)
        f.eksportDoRSES(attributes_info, reguly, nazwa_tabeli, f"hierarchiczne_{nazwa_tabeli}_grupy{i}.tab")

#################################################

# --- Grupowanie DBSCAN ---

def grupowanie3_DBSCAN(features, nazwa_tabeli, attributes_info):
    min_samples = features.shape[1] * 2
    f.wykresNajblizszychOdleglosci(features, min_samples)
    eps = 5.0
    clustersDBSCAN = f.grupowanieDBSCAN(features, eps, min_samples)
    grupyDBSCAN = f.przypisanieGrup(features, clustersDBSCAN)
    reguly = f.regulyDecyzyjne(grupyDBSCAN)
    f.eksportDoRSES(attributes_info, reguly, nazwa_tabeli, f"DBSCAN_{nazwa_tabeli}.tab")

##################################################



