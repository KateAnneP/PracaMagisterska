import utils as f

#Zrób do tabeli zestawienie wyników wszystkich miar jakości dla wszystkich grupowań
#Dopisz o doborze eps i min_samples do teorii, ale nie masz źródła
#Główny algorytm tylko jako pseudokod, a resztę jako algorytmy??? ale wpierdziel je do teorii
#... algorytm wybrał ilość grup taką i taką, a z powyższych danych wynika, że jakość tego grupowania jest taka i taka
#PCA i drzewo decyzyjne dołóż

def grupowanie(features, nazwa_tabeli, attributes_info, nr_grupowania):
    #Nr grupowania: 1-Kmeans, 2-hierarchiczne, 3-DBSCAN
    if nr_grupowania == 1:
        nazwa_grupowania = "Kmeans"
    elif nr_grupowania == 2:
        nazwa_grupowania = "hierarchiczne"
    elif nr_grupowania == 3:
        nazwa_grupowania = "DBSCAN"

    allClusters = []
    if nr_grupowania == 1:
        centroids_Kmeans = []
        inertia_Kmeans = []

    #Tablice używane do wykresów
    silhouettes = []
    dbi = []
    calinski_harabasz = []

    #Słowniki używane do właściwej ilości grup
    best_silhouette = {0: -1}
    best_dbi = {0: -1}
    best_ch = {0: -1}
    best_indexes = []

    for i in range(2,10):

        if nr_grupowania == 1:
            clusters, centroids, inertia = f.grupowanieKmeans(i, features)
            allClusters.append(clusters)
            centroids_Kmeans.append(centroids)
            inertia_Kmeans.append(inertia)
        elif nr_grupowania == 2:
            clusters = f.grupowanieHierarchiczne(i, features, "complete", "euclidean")
            allClusters.append(clusters)
        elif nr_grupowania == 3:
            min_samples = features.shape[1] * 2
            f.wykresNajblizszychOdleglosci(features, min_samples)
            eps = 5.0
            clustersDBSCAN = f.grupowanieDBSCAN(features, eps, min_samples)
            grupyDBSCAN = f.przypisanieGrup(features, clustersDBSCAN)
            reguly = f.regulyDecyzyjne(grupyDBSCAN)
            f.eksportDoRSES(attributes_info, reguly, nazwa_tabeli, f"DBSCAN_{nazwa_tabeli}.tab")
            break

        #Wyszukiwanie najlepszej wartości miary Silhouette
        current_silhouette = f.miaraSilhouette(features, clusters)
        best_silhouette = list(best_silhouette.values())[0]
        if current_silhouette >= best_silhouette:
            best_silhouette.clear()
            best_silhouette[i] = current_silhouette

        #Wyszukiwanie najlepszej wartości miary DBI
        current_dbi = f.miaraDBI(features, clusters)
        best_dbi = list(best_dbi.values())[0]
        if current_silhouette >= best_dbi:
            best_dbi.clear()
            best_dbi[i] = current_dbi

        # Wyszukiwanie najlepszej wartości miary DBI
        current_ch = f.miaraCalinskiHarabasz(features, clusters)
        best_ch = list(best_ch.values())[0]
        if current_ch >= best_ch:
            best_ch.clear()
            best_ch[i] = current_ch

        silhouettes.append(current_silhouette)
        dbi.append(current_dbi)
        calinski_harabasz.append(current_ch)

    # Wykresy miar jakości grupowania
    if nr_grupowania == 1:
        f.pokazWykresLokcia(inertia_Kmeans, 10)
    f.pokazWykresSilhouette(silhouettes, 10)
    f.pokazWykresDBI(dbi, 10)
    f.pokazWykresCalinskiHarabasz(calinski_harabasz, 10)

    best_indexes_Kmeans = [list(best_silhouette.keys())[0], list(best_dbi.keys())[0], list(best_ch.keys())[0]]
    print(f"Na ile grup powinny być podzielone dane w grupowaniu {nazwa_grupowania} wg miar jakości grupowania: {best_indexes}")

    #Dla każdej liczby grup, która jest najlepsza wg miar jakości wykonywane jest przypisanie grup do danych, a potem stworzenie reguł decyzyjnych i eksport do RSES
    for i in dict.fromkeys(best_indexes_Kmeans):
        clustersG = allClusters[i]
        grupy = f.przypisanieGrup(features, clustersG)
        reguly = f.regulyDecyzyjne(grupy)
        f.eksportDoRSES(attributes_info, reguly, nazwa_tabeli, f"{nazwa_grupowania}_{nazwa_tabeli}_grupy{i}.tab")


#########################################
