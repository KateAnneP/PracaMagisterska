import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import utils as f

# Nie masz podkładki pod wzory na a i b w mierze Silhouette, dorób może rysunek do tego, jak wyglądają te grupy A i C
# Dodaj parę fotek do algorytmów grupowania
# !!! Popraw algorytm, z tymi warunkami dot. miar jakości, że większe od 0

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


    features_copy = features.copy()
    features_scaled = MinMaxScaler().fit_transform(features_copy)
    features_df = pd.DataFrame(features_scaled, columns=features.columns, index=features.index)

    # for i in range(2,3):
    i = 2
    if nr_grupowania == 1:
        clusters, centroids, inertia = f.grupowanieKmeans(i, features_df)
        allClusters.append(clusters)
        centroids_Kmeans.append(centroids)
        inertia_Kmeans.append(inertia)
    elif nr_grupowania == 2:
        clusters = f.grupowanieHierarchiczne(i, features_df, "complete", "euclidean")
        allClusters.append(clusters)
    elif nr_grupowania == 3:
        min_samples = features_df.shape[1] * 2
        f.wykresNajblizszychOdleglosci(features, min_samples)
        eps = 70.0
        clustersDBSCAN = f.grupowanieDBSCAN(features_df, eps, min_samples)
        grupyDBSCAN = f.przypisanieGrup(features, clustersDBSCAN)
        f.wykresPCA(features, clustersDBSCAN, nazwa_grupowania)
        reguly = f.regulyDecyzyjne(grupyDBSCAN)
        #f.eksportDoRSES(attributes_info, reguly, nazwa_tabeli, f"DBSCAN_{nazwa_tabeli}.tab")
        return grupyDBSCAN

    # f.wykresPCA(features, allClusters[0], nazwa_grupowania)
    # f.wykresPCA(features, allClusters[1], nazwa_grupowania)


    clustersG = allClusters[0]
    X_train, X_test, y_train, y_test = train_test_split(features, clustersG, train_size=0.5, shuffle=True, random_state=42 )

    grupy_train = f.przypisanieGrup(X_train, y_train)
    grupy_test = f.przypisanieGrup(X_test, y_test)
    #grupy = f.przypisanieGrup(features, clustersG)
    reguly_train = f.regulyDecyzyjne(grupy_train)
    reguly_test = f.regulyDecyzyjne(grupy_test)
    #reguly_caly = f.regulyDecyzyjne(grupy)

    f.eksportDoRSES(attributes_info, reguly_train, nazwa_tabeli, f"wyniki/{nazwa_grupowania}_{nazwa_tabeli}_grupy{i}.tab")
    #reguly_train.to_csv(f"wyniki/{nazwa_grupowania}_{nazwa_tabeli}_grupy{i}.csv", index=False) #zmienione na i + 2

    return reguly_test, reguly_train #clusters

def count_matching_rows(df: pd.DataFrame, query: dict) -> int:
    mask = pd.Series(True, index=df.index)
    for col, val in query.items():
        mask &= df[col] == val
    return mask.sum()

def stabilnosc(reguly_test, reguly_train):
    klawisz = 1
    while(klawisz == 1):
        cechy = {}
        n = int(input("Podaj liczbę cech do zweryfikowania: "))
        for i in range(0,n):
            cecha = input("Podaj cechę: ")
            cecha = 'feature_' + cecha
            values = input("Podaj wartosci: ")
            cechy[cecha] = '"' + values + '"'

        print(cechy)

        #Dla test:
        wiersze_spelniajace_czesc_warunkowa = count_matching_rows(reguly_test, cechy)
        #print(f"Wiersze spełniające część warunkową: {wiersze_spelniajace_czesc_warunkowa}")
        wielkosc_zbioru = len(reguly_test)
        pokrycie = wiersze_spelniajace_czesc_warunkowa/wielkosc_zbioru
        print(f"Pokrycie test: {round(pokrycie, 3)}")

        # Dla train:
        wiersze_spelniajace_czesc_warunkowa_train = count_matching_rows(reguly_train, cechy)
        #print(f"Wiersze spełniające część warunkową cały zbiór: {wiersze_spelniajace_czesc_warunkowa_train}")
        wielkosc_zbioru_train = len(reguly_train)
        pokrycie_train = wiersze_spelniajace_czesc_warunkowa_train / wielkosc_zbioru_train
        #print(f"Pokrycie train: {pokrycie_train}")

        # DLa test:
        grupa = input("Podaj zmianę grupy: ")
        cechy['group'] = '"' + grupa + '"'
        wiersze_spelniajace_czesc_warunkowa_i_decyzyjna = count_matching_rows(reguly_test, cechy)
        #print(f"Wiersze spełniające część warunkową i decyzyjną: {wiersze_spelniajace_czesc_warunkowa_i_decyzyjna}")
        precyzja = wiersze_spelniajace_czesc_warunkowa_i_decyzyjna/wiersze_spelniajace_czesc_warunkowa
        print(f"Precyzja test: {round(precyzja,3)}")

        # Dla train:
        wiersze_spelniajace_czesc_warunkowa_i_decyzyjna_train = count_matching_rows(reguly_train, cechy)
        #print(f"Wiersze spełniające część warunkową i decyzyjną cały: {wiersze_spelniajace_czesc_warunkowa_i_decyzyjna_train}")
        precyzja_train = wiersze_spelniajace_czesc_warunkowa_i_decyzyjna_train / wiersze_spelniajace_czesc_warunkowa_train
        #print(f"Precyzja train: {precyzja_train}")

        prec = precyzja_train - precyzja
        stabilnosc_precyzja = 1 - abs(prec)
        stabilnosc_pokrycie = 1 - abs(pokrycie_train - pokrycie)
        print(f"Stabilnosc(pokrycie) = {round(stabilnosc_pokrycie,3)},\nStabilność(precyzja) = {round(stabilnosc_precyzja,3)}\n")

        klawisz = int(input("Podaj klawisz (1 - licz dalej, 0 - przerwij): "))



#########################################

def wykresy_i_miary(clusters, features, nr_grupowania, nazwa_grupowania):
    # Tablice używane do wykresów
    silhouettes = []
    dbi = []
    calinski_harabasz = []

    # Słowniki używane do właściwej ilości grup
    best_silhouette = {0: -1}
    best_dbi = {0: 100}
    best_ch = {0: -1}
    best_indexes = []

    clusters = []
    centroids = []
    inertia = []

    miary = {}

    #Wyszukiwanie najlepszej wartości miary Silhouette
    current_silhouette = f.miaraSilhouette(features, clusters)
    best_silhouette_value = list(best_silhouette.values())[0]
    if current_silhouette >= best_silhouette_value and current_silhouette >= 0:
        best_silhouette.clear()
        best_silhouette[i] = current_silhouette

    #Wyszukiwanie najlepszej wartości miary DBI
    current_dbi = f.miaraDBI(features, clusters)
    best_dbi_value = list(best_dbi.values())[0]
    if current_dbi <= best_dbi_value:
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

    #Wykresy miar jakości grupowania
    if nr_grupowania == 1:
        f.pokazWykresLokcia(inertia_Kmeans, 10)
    f.pokazWykresSilhouette(silhouettes, 10)
    f.pokazWykresDBI(dbi, 10)
    f.pokazWykresCalinskiHarabasz(calinski_harabasz, 10)

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
    for i in dict.fromkeys(best_indexes):
        pass


#########################################
def drzewoDecyzyjne(clusters, features):
    # noColumn = dane.shape[1]
    # features = dane.iloc[:, :noColumn - 1]
    features_names = list(features)
    labels = clusters

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

    # text_representation = tree.export_text(model, feature_names=features_names, class_names=my_class_names, )
    # print("Wizualizacja tekstowa:")
    # print(text_representation)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
    plot_tree(model, feature_names=features_names, class_names=my_class_names, rounded=True, filled=True,
              proportion=True)
    plt.show()

    # accuracy = metrics.accuracy_score(labels_test, labels_predicted)
    # print(f"Dokładność klasyfikacji: {accuracy}")
    # print("========= PEŁNE WYNIKI KLASYFIKACJI ================")
    # report = classification_report(labels_test, labels_predicted)
    # print(report)
    # print("====== MACIERZ POMYŁEK (confusion matrix) +=========")
    # conf_matrix = confusion_matrix(labels_test, labels_predicted)
    # print(conf_matrix)



