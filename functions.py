import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics, tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

import utils as f

#Zrób do tabeli zestawienie wyników wszystkich miar jakości dla wszystkich grupowań
#... algorytm wybrał ilość grup taką i taką, a z powyższych danych wynika, że jakość tego grupowania jest taka i taka
#Dopisz do eksportu do RSESa (w dodatkach)" że tam atrybuty trzeba, i że rozszerzenie itp.
#Dodać kolejny algorytm, w którym są wyniki z testów dla obu małżonków.

# Nie masz podkładki pod wzory na a i b w mierze Silhouette, dorób może rysunek do tego, jak wyglądają te grupy A i C
# Metryki odległości - zrób eksperymenty, czy coś będzie lepsze od euklidesowej, jak coś to dopisz
# Dodaj parę fotek do algorytmów grupowania

#Pytania:
# - czy do łokciowej może być od 2, bo brzydko wygląda wykres
# - czy do atrybutów dajemy info o userze, skoro tego się nie da zmienić? (Ale w sumie może wyjść wpływ, że np. w starszym wieku bardziej są ludzie podatni)
# - #Czy klasyfikować ich jako pojedyncze osoby, ale z partnerem, czy jako pary.
# #Bo jeśli  jako pary, to płeć można całkiem usunąć, skoro obie osoby są brane pod uwagę
# #Ale jeśli jako pary, to czyją chorobę mamy na myśli. Chyba, że chodzi tylko o sam fakt jej istnienia
# #Więc chyba można wziąć pod uwagę pojedyncze osoby, ale z uwzględnieniem wieku i edukacji ich partnera, żeby klasyfikować ich zdrowie psychiczne
# # w zależności od ich relacji

def grupowanie(features, nazwa_tabeli, attributes_info, nr_grupowania):
    #Nr grupowania: 1-Kmeans, 2-hierarchiczne, 3-DBSCAN
    if nr_grupowania == 1:
        nazwa_grupowania = "Kmeans"
    elif nr_grupowania == 2:
        nazwa_grupowania = "hierarchiczne"
    elif nr_grupowania == 3:
        nazwa_grupowania = "DBSCAN"

    allClusters = []
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

    clusters = []
    centroids = []
    inertia = []

    miary = {}

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
            return grupyDBSCAN

        #Wyszukiwanie najlepszej wartości miary Silhouette
        current_silhouette = f.miaraSilhouette(features, clusters)
        best_silhouette_value = list(best_silhouette.values())[0]
        if current_silhouette >= best_silhouette_value:
            best_silhouette.clear()
            best_silhouette[i] = current_silhouette

        #Wyszukiwanie najlepszej wartości miary DBI
        current_dbi = f.miaraDBI(features, clusters)
        best_dbi_value = list(best_dbi.values())[0]
        if current_silhouette >= best_dbi_value:
            best_dbi.clear()
            best_dbi[i] = current_dbi

        # Wyszukiwanie najlepszej wartości miary DBI
        current_ch = f.miaraCalinskiHarabasz(features, clusters)
        best_ch_value = list(best_ch.values())[0]
        if current_ch >= best_ch_value:
            best_ch.clear()
            best_ch[i] = current_ch

        silhouettes.append(current_silhouette)
        dbi.append(current_dbi)
        calinski_harabasz.append(current_ch)

    # Wykresy miar jakości grupowania
    if nr_grupowania == 1:
        f.pokazWykresLokcia(inertia_Kmeans, 10)
        f.wykresPCA(features, clusters, nazwa_grupowania, centroids)
    else:
        f.wykresPCA(features, clusters, nazwa_grupowania)
    # f.pokazWykresSilhouette(silhouettes, 10)
    # f.pokazWykresDBI(dbi, 10)
    # f.pokazWykresCalinskiHarabasz(calinski_harabasz, 10)

    miary['silhouette'] = silhouettes
    miary['dbi'] = dbi
    miary['calinski_harabasz'] = calinski_harabasz
    import pandas as pd
    df = pd.DataFrame({
        'Liczba klastrów': range(2, 2 + len(miary['silhouette'])),
        'Silhouette': miary['silhouette'],
        'DBI': miary['dbi'],
        'Calinski-Harabasz': miary['calinski_harabasz']
    })

    print(df.to_string(index=False))

    best_indexes = [list(best_silhouette.keys())[0], list(best_dbi.keys())[0], list(best_ch.keys())[0]]
    print(f"Na ile grup powinny być podzielone dane w grupowaniu {nazwa_grupowania} wg miar jakości grupowania: {best_indexes}")

    #Dla każdej liczby grup, która jest najlepsza wg miar jakości wykonywane jest przypisanie grup do danych, a potem stworzenie reguł decyzyjnych i eksport do RSES
    # for i in dict.fromkeys(best_indexes):
    #     clustersG = allClusters[i]
    #     grupy = f.przypisanieGrup(features, clustersG)
    #     reguly = f.regulyDecyzyjne(grupy)
    #     reguly.to_csv(f'{nazwa_grupowania}_{nazwa_tabeli}_grupy{i}.csv', index=False)
    #     f.eksportDoRSES(attributes_info, reguly, nazwa_tabeli, f"{nazwa_grupowania}_{nazwa_tabeli}_grupy{i}.tab")
    #
    return #grupy

#########################################

def drzewoDecyzyjne(dane):
    noColumn = dane.shape[1]
    features = dane.iloc[:, :noColumn - 1]
    features_names = list(features)
    labels = dane.iloc[:, [noColumn - 1]]

    datasets = train_test_split(features, labels, test_size=0.3, random_state=1234) #Dla test_size = 0.4 i 0.2 dokładność jest niższa - optymalna dla 0.3
    features_train = datasets[0]
    features_test = datasets[1]
    labels_train = datasets[2]
    labels_test = datasets[3]

    # Parametry drzewa
    model = DecisionTreeClassifier(criterion='gini',
                                   max_depth=5,
                                   min_samples_split=10,
                                   min_samples_leaf=10,
                                   max_leaf_nodes=30,
                                   min_impurity_decrease=0.02)

    model.fit(features_train, np.ravel(labels_train))
    labels_predicted = model.predict(features_test)

    class_names_ordered = sorted(np.unique(labels))
    my_class_names = []
    for i in range(0, len(class_names_ordered)):
        my_class_names.append(str(class_names_ordered[i]))

    text_representation = tree.export_text(model, feature_names=features_names, class_names=my_class_names, )
    print("Wizualizacja tekstowa:")
    print(text_representation)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 5), dpi=600)
    plot_tree(model, feature_names=features_names, class_names=my_class_names, rounded=True, filled=True,
              proportion=True);

    accuracy = metrics.accuracy_score(labels_test, labels_predicted)
    print(f"Dokładność klasyfikacji: {accuracy}")
    print("========= PEŁNE WYNIKI KLASYFIKACJI ================")
    report = classification_report(labels_test, labels_predicted)
    print(report)
    print("====== MACIERZ POMYŁEK (confusion matrix) +=========")
    conf_matrix = confusion_matrix(labels_test, labels_predicted)
    print(conf_matrix)