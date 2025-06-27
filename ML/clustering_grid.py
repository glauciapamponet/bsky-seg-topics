#%%
import s3fs
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from itertools import product

from sklearn.cluster import KMeans, HDBSCAN, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from Assets.Code import LoadingData

data_loader = LoadingData.LoadingData()

mlflow.set_tracking_uri("http://127.0.0.1:5000/")

#%%

def fit_model(model, vectors):
    model.fit(vectors)
    labels = model.labels_
    mask = labels != -1
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    try:
        metrics = dict(
            silhouette_avg = silhouette_score(vectors[mask], labels[mask]),
            davies_bouldin = davies_bouldin_score(vectors[mask], labels[mask]),
            calinski_harabasz = calinski_harabasz_score(vectors[mask], labels[mask])
        )
    except ValueError as e:
        metrics = {"silhouette_avg" : -1, "davies_bouldin" : -1, "calinski_harabasz" : -1}

    return metrics, n_clusters

#%%
def elbow_dbscan(samples_values, vec):
    plt.figure(figsize=(10, 6))

    for k in samples_values:
        neighbors_fit = NearestNeighbors(n_neighbors=k).fit(vec)  
        distances, _ = neighbors_fit.kneighbors(vec)

        kth_distances = np.sort(distances[:, k - 1])
        plt.plot(kth_distances, label=f'k = {k}')

    plt.xlabel("Pontos ordenados")
    plt.ylabel(f"Distância até o {k}-ésimo vizinho")
    plt.title("Gráfico do Cotovelo para escolha de eps (vários sample_values)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def elbow_kmeans(k_range, vec):
    inertias = []
    for k in range(1, k_range):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(vec)
        inertias.append(kmeans.inertia_)

    plt.plot(range(1, k_range), inertias, 'o-', color='blue')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inércia (Distância intra-cluster)')
    plt.title('Método do Cotovelo')
    plt.grid(True)
    plt.show()

def cleaning_similarity(X):
    similarity_matrix = cosine_similarity(X)

    threshold = 0.95
    to_drop = set()

    for i in range(similarity_matrix.shape[0]):
        if i in to_drop:
            continue
        for j in range(i + 1, similarity_matrix.shape[1]):
            if similarity_matrix[i, j] > threshold:
                to_drop.add(j)
    
    return to_drop

# %%
df_posts = data_loader.load_posts()

X  = np.vstack(df_posts["embedding"].values)
not_in_set = np.setdiff1d(np.arange(X.shape[0]), list(cleaning_similarity(X)))

X_minMax = MinMaxScaler().fit_transform(X[not_in_set])
X_robust = RobustScaler().fit_transform(X[not_in_set])
#%%
elbow_dbscan([50, 80, 110, 140], X_minMax)
elbow_dbscan([20, 50, 80], X_robust)

elbow_kmeans(10, X_minMax)
elbow_kmeans(10, X_robust)

# %% Grid Search DBSCAN

samples = [50, 80, 110, 140]
eps_values = [2.4, 2.5, 2.6]
distances = ['euclidean', 'cosine']
vecs = {"minMax": X_minMax, "robust": X_robust}
combinations = list(product(eps_values, samples, distances, vecs.keys()))
i = 1

for eps, sample, dist, vec in combinations:
    mlflow.set_experiment(experiment_id=818640941147033185)
    with mlflow.start_run():
        clt = DBSCAN(eps=eps, 
                        min_samples=sample, 
                        metric=dist, 
                        n_jobs=6)
        metrics, n_clusters = fit_model(clt, vecs[vec])
    
        mlflow.set_tag("mlflow.runName", f"DBSCAN-SR-POS-SIMILARITY-{i}")
        mlflow.log_params({"eps": eps,
                            "min_samples": sample,
                            "distance": dist,
                            "vector": vec,
                            "n_clusters": n_clusters})
        mlflow.log_metrics(metrics)
    i+=1


#%% Grid Search HDBSCAN

samples = [50, 80, 110, 140]
sizes = [30, 50, 80]
distances = ['cosine']
vecs = {"Robust": X_robust, "minMax": X_minMax}
combinations = list(product(distances, samples, sizes, vecs.keys()))
i = 1
for distance, sample, size, vec in combinations:
    mlflow.set_experiment(experiment_id=114565549749737657)
    with mlflow.start_run():
        clt = HDBSCAN(min_cluster_size=size, 
                        min_samples=sample, 
                        metric=distance, 
                        cluster_selection_method="eom",
                        n_jobs=6)
        metrics, n_clusters = fit_model(clt, vecs[vec])

        mlflow.set_tag("mlflow.runName", f"HDBSCAN-SR-POS-SIMILARITY-{i}")
        mlflow.log_params({"vector": vec,
                        "cluster_size": size,
                        "min_samples": sample,
                        "metric": distance,
                        "method": "eom",
                        "n_clusters": n_clusters})
        mlflow.log_metrics(metrics)
    i+=1
        
#%% Grid Search KMeans

n_clusters = [3, 4, 5]
n_inits = [10, 30, 50, 80]
inits = ['k-means++', 'random']
vecs = {"Robust": X_robust, "minMax": X_minMax}
combinations = list(product(n_clusters, inits, n_inits, vecs.keys()))
i = 1

for cluster, init, n_init in combinations:
    mlflow.set_experiment(experiment_id=975870007803510489)
    with mlflow.start_run():     
        clt = KMeans(n_clusters=cluster,
                        init=init,
                        n_init=n_init,
                        random_state=42)
        metrics, n_clusters = fit_model(clt, vecs[vec])

        mlflow.set_tag("mlflow.runName", f"Kmeans-{i}")
        mlflow.log_params({
            "vector": vec,
            "n_init": n_init,
            "init": init,
            "n_clusters": cluster})
        mlflow.log_metrics(metrics)
    i+=1
