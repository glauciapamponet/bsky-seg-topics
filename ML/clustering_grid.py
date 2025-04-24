#%%
import s3fs
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from itertools import product

from sklearn.cluster import KMeans, HDBSCAN, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


BUCKET_PATH = "s3://bsky-posts-lake"
GOLD_POSTS_PATH = f"{BUCKET_PATH}/gold/fato/posts"
SILVER_POSTS_PATH = f"{BUCKET_PATH}/silver/fato/posts"

mlflow.set_tracking_uri("http://127.0.0.1:5000/")

#%%
def loading_table(path):
    s3 = s3fs.S3FileSystem()
    parquet_files = s3.glob(path)
    df_posts = pd.DataFrame()
    for file in parquet_files:
        with s3.open(file) as f:
            df_posts = pd.concat([df_posts, pd.read_parquet(f)])
    return df_posts

def fit_model(model, vectors):
    model.fit(vectors)
    labels = model.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    try:
        metrics = dict(
            silhouette_avg = silhouette_score(vecs[vec], labels),
            davies_bouldin = davies_bouldin_score(vecs[vec], labels),
            calinski_harabasz = calinski_harabasz_score(vecs[vec], labels)
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

# %%
df_posts = loading_table(f"{GOLD_POSTS_PATH}/*/*.parquet")

X  = np.vstack(df_posts["embedding"].values)
X_minMax = MinMaxScaler().fit_transform(X)
X_robust = RobustScaler().fit_transform(X)
#%%
elbow_dbscan([20, 50, 80], X_minMax)
elbow_dbscan([20, 50, 80], X_robust)

elbow_kmeans(10, X_minMax)
elbow_kmeans(10, X_robust)

# %%
# Grid Search DBSCAN

samples = [20, 50, 80]
eps_values = [0.8, 1.1, 1.4]
distances = ['cosine', 'euclidean']
vecs = {"Robust": X_robust, "minMax": X_minMax}
combinations = list(product(eps_values, samples, distances, vecs.keys()))
i = 1

for eps, sample, dist, vec in combinations:
    mlflow.set_experiment(experiment_id=931016321717885447)
    with mlflow.start_run():
        clt = DBSCAN(eps=eps, 
                        min_samples=sample, 
                        metric=dist, 
                        n_jobs=6)
        metrics, n_clusters = fit_model(clt, vecs[vec])
    
        mlflow.set_tag("mlflow.runName", f"DBSCAN-{i}")
        mlflow.log_params({"eps": eps,
                            "min_samples": sample,
                            "n_clusters": n_clusters})
        mlflow.log_metrics(metrics)
    i+=1


#%% 
# Grid Search HDBSCAN

samples = [20, 50, 80]
sizes = [15, 30, 50]
distances = ['cosine', 'euclidean']
vecs = {"Robust": X_robust, "minMax": X_minMax}
combinations = list(product(samples, distances, sizes, vecs.keys()))
i = 1

for distance, sample, size, vec in combinations:
    mlflow.set_experiment(experiment_id=445386104313139070)
    with mlflow.start_run():
        clt = HDBSCAN(min_cluster_size=size, 
                        min_samples=sample, 
                        metric=distance, 
                        cluster_selection_method="eom",
                        n_jobs=6)
        metrics, n_clusters = fit_model(clt, vecs[vec])

        mlflow.set_tag("mlflow.runName", f"HDBSCAN-{i}")
        mlflow.log_params({"vector": vec,
                        "cluster_size": size,
                        "min_samples": sample,
                        "metric": distance,
                        "method": "eom",
                        "n_clusters": n_clusters})
        mlflow.log_metrics(metrics)
    i+=1
        
#%%
n_clusters = [3, 4, 5]
n_inits = [10, 30, 50, 80]
inits = ['k-means++', 'random']
vecs = {"Robust": X_robust, "minMax": X_minMax}
combinations = list(product(n_clusters, inits, n_inits, vecs.keys()))
i = 53

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
