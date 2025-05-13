import numpy as np
import pandas as pd
import matplotlib
from matplotlib.lines import Line2D
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import scipy.cluster.hierarchy as shc
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors

def pokazWykresLokcia(results, n):
    fig = plt.figure(figsize=(10, 7))
    plt.plot(range(2, n), results, marker='o')
    plt.title("Optymalizacja skupień metodą łokcia")
    plt.xlabel('Liczba skupień')
    plt.ylabel('Miara niespójności')
    plt.tight_layout()
    plt.show()

def miaraSilhouette(features, clusters):
    silhouette = silhouette_score(features, clusters)
    return silhouette

def pokazWykresSilhouette(results, n):
    fig = plt.figure(figsize=(10, 7))
    plt.plot(range(2, n), results, marker='o')
    plt.title("Wykres jakości grupowania metodą Silhouette")
    plt.xlabel('Liczba skupień')
    plt.ylabel('Wartosć')
    plt.tight_layout()
    plt.show()

def miaraDBI(features, clusters):
    dbi = davies_bouldin_score(features, clusters)
    return dbi

def pokazWykresDBI(results, n):
    fig = plt.figure(figsize=(10, 7))
    plt.plot(range(2, n), results, marker='o')
    plt.title("Wykres jakości grupowania metodą Davies - Bouldin")
    plt.xlabel('Liczba skupień')
    plt.ylabel('Wartosć')
    plt.tight_layout()
    plt.show()

def miaraCalinskiHarabasz(features, clusters):
    ch_score = calinski_harabasz_score(features, clusters)
    return ch_score

def pokazWykresCalinskiHarabasz(results, n):
    fig = plt.figure(figsize=(10, 7))
    plt.plot(range(2, n), results, marker='o')
    plt.title("Wykres jakości grupowania metodą Calinski-Harabasz")
    plt.xlabel('Liczba skupień')
    plt.ylabel('Wartosć')
    plt.tight_layout()
    plt.show()

def grupowanieKmeans(k, features):
    # Utworzenie obiektu do grupowania
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=0)
    kmeans.fit(features)
    clusters = kmeans.fit_predict(features)
    inertia = kmeans.inertia_ #Miara jakości
    centroids = kmeans.cluster_centers_ #Centroidy
    return clusters, centroids, inertia

def narysujDendrogram(features, method='ward'):
    plt.figure(figsize=(12, 8))
    plt.title("Dendrogram")
    shc.dendrogram(shc.linkage(features, method=method))
    plt.show()

def grupowanieHierarchiczne(k, features, linkage, metric):
    ac = AgglomerativeClustering(n_clusters=k, metric=metric, linkage=linkage)
    clusters = ac.fit_predict(features)
    return clusters

def wykresNajblizszychOdleglosci(features, min_samples):
    neigh = NearestNeighbors(n_neighbors=min_samples)
    neigh.fit(features)
    dists, _ = neigh.kneighbors(features)
    odleglosci = np.sort(dists[:, -1])  # bierzemy ostatnią kolumnę (najdalszy z k)
    plt.plot(odleglosci)
    plt.xlabel("Punkty")
    plt.ylabel(f"{min_samples}-najbliższa odległość")
    plt.title("Wykres k-odległości")
    plt.grid(True)
    plt.show()

def grupowanieDBSCAN(features, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = model.fit_predict(features)
    return clusters

def wykresPCA(features, clusters, nazwa, centroids=None):
    # Redukcja wymiarów za pomocą PCA - dane są sprowadzane do 2 wymiarów
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    if centroids is not None:
        centroids_pca = pca.transform(centroids)

    plt.figure(figsize=(10, 7))
    plt.title(f'Wizualizacja - grupowanie {nazwa}')
    #plt.scatter(features_pca[:, 0], features_pca[:, 1], c=clusters, cmap='tab10', s=10, alpha=0.7)
    if centroids is not None:
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=100, color='blue', marker='x')

    #Unikalne klastry i kolory
    unique_clusters = np.unique(clusters)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
    color_map = dict(zip(unique_clusters, colors))

    #Punkty z kolorami zgodnymi z mapą
    for cluster in unique_clusters:
        idx = clusters == cluster
        plt.scatter(features_pca[idx, 0], features_pca[idx, 1], color=color_map[cluster], s=10, alpha=0.7, label=f'Klaster {cluster}')

    plt.xlabel('Składowa główna 1')
    plt.ylabel('Składowa główna 2')
    plt.legend()
    plt.show()

def przypisanieGrup(dane, clusters):
    noColumn = dane.shape[1]
    print(noColumn) #Sprawdź to
    new_column = pd.Series(clusters, name='group')
    dane = pd.concat([dane, new_column], axis=1)
    wiersze = []

    print(dane)

    for i in range(0, len(dane) - 1):
        for j in range(0, len(dane) - 1):
            if i != j:
                index_i = pd.Series(dane.index[i], index=['index_i'])
                features_i = dane.iloc[i, 0:noColumn].reset_index(drop=True)
                features_i.index = [f'feature_{k}_i' for k in range(0, len(features_i))]
                group_i = pd.Series(dane.iloc[i, noColumn], index=['group_i'])
                index_j = pd.Series(dane.index[j], index=['index_j'])
                features_j = dane.iloc[j, 0:noColumn].reset_index(drop=True)
                features_j.index = [f'feature_{k}_j' for k in range(0, len(features_j))]
                group_j = pd.Series(dane.iloc[j, noColumn], index=['group_j'])
                wiersz = pd.concat([features_i, features_j, index_i, group_i, index_j, group_j], axis=0)
                wiersze.append(wiersz)

    zestawienie = pd.DataFrame(wiersze)
    return zestawienie

def regulyDecyzyjne(zestawienie, nazwa_tabeli):
    rows = []

    # if(nazwa_tabeli == "anxiety" or nazwa_tabeli=="depression"):
    #     for i in range(0, len(zestawienie)):
    #         row = zestawienie.iloc[i]
    #         if int(row['group_i']) != int(row['group_j']):  # wykluczenie wierszy, w których nie ma zmiany grupy
    #             wiersz = {
    #                 'feature_0': f"\"{row['feature_0_i']}-{row['feature_0_j']}\"",
    #                 'feature_1': f"\"{row['feature_1_i']}-{row['feature_1_j']}\"",
    #                 'feature_2': f"\"{row['feature_2_i']}-{row['feature_2_j']}\"",
    #                 'feature_3': f"\"{row['feature_3_i']}-{row['feature_3_j']}\"",
    #                 'feature_4': f"\"{row['feature_4_i']}-{row['feature_4_j']}\"",
    #                 'feature_5': f"\"{row['feature_5_i']}-{row['feature_5_j']}\"",
    #                 'feature_6': f"\"{row['feature_6_i']}-{row['feature_6_j']}\"",
    #                 'feature_7': f"\"{row['feature_7_i']}-{row['feature_7_j']}\"",
    #                 'feature_8': f"\"{row['feature_8_i']}-{row['feature_8_j']}\"",
    #                 'feature_9': f"\"{row['feature_9_i']}-{row['feature_9_j']}\"",
    #                 'group': f"\"{row['group_i']}-{row['group_j']}\""
    #             }
    #             rows.append(wiersz)
    # elif(nazwa_tabeli == "anxiety_depression" or nazwa_tabeli == "anxiety_user" or nazwa_tabeli == "depression_user"):
    noColumn = int(zestawienie.shape[1]/2)-2
    print(noColumn)
    for i in range(0, len(zestawienie)):
        row = zestawienie.iloc[i]
        if int(row['group_i']) != int(row['group_j']):
            wiersz = {
                f"feature_{i}": f"\"{row[f'feature_{i}_i']}-{row[f'feature_{i}_j']}\""
                for i in range(noColumn)
            }
            wiersz['group'] = f"\"{row['group_i']}-{row['group_j']}\""
            rows.append(wiersz)
    # elif(nazwa_tabeli == "anxiety_depression_user"):
    #     for i in range(0, len(zestawienie)):
    #         row = zestawienie.iloc[i]
    #         if int(row['group_i']) != int(row['group_j']):
    #             wiersz = {
    #                 f"feature_{i}": f"\"{row[f'feature_{i}_i']}-{row[f'feature_{i}_j']}\""
    #                 for i in range(32)
    #             }
    #             wiersz['group'] = f"\"{row['group_i']}-{row['group_j']}\""
    #             rows.append(wiersz)
    # elif(nazwa_tabeli == "anxiety_depression_pairs"):
    #     pass
    reguly = pd.DataFrame(rows)
    return reguly

def eksportDoRSES(attributes_info, df, table_name, filename):
    """
       Eksportuje DataFrame do pliku w formacie tabelarycznym (tab).

       :param df: Pandas DataFrame z danymi.
       :param attributes_info: Lista krotek (nazwa_atrybutu, typ, [precision]).
       :param table_name: Nazwa tabeli.
       :param filename: Nazwa pliku wyjściowego.
       """
    with open(filename, 'w') as f:
        # Nagłówek
        f.write(f'TABLE "{table_name}"\n')
        f.write(f'ATTRIBUTES {len(attributes_info)}\n')
        for attr in attributes_info:
            line = f' {attr[0]} {attr[1]}'
            if len(attr) > 2:  # Jeśli podano precyzję
                line += f' {attr[2]}'
            f.write(line + '\n')
        f.write(f'OBJECTS {len(df)}\n')

        # Dane obiektów
        for _, row in df.iterrows():
            f.write(" ".join(map(str, row.values)) + '\n')

    print(f"Eksport zakończony. Plik został zapisany jako {filename}.")