#%%
import random
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.cluster import HDBSCAN, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score, pairwise_distances
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.pipeline import Pipeline

from itertools import combinations

from Assets.Code import LoadingData

data_loader = LoadingData.LoadingData()

PLOT_MLFLOW_ARTIFACTS = f"../Assets/Mlflow Artifacts"
DATA_MLFLOW_ARTIFACTS = f"../Data"
EXPERIMENT_ID = 793754980848381686

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
#%% MODEL CLASS
def gerar_palette_clusters(cluster_set):
    def cor_aleatoria_existente(existentes):
        cor = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        while cor in existentes or cor == "#BCBCBC":
            cor = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        return cor

    usadas = set()
    palette = {}

    for i in cluster_set:
        cor = cor_aleatoria_existente(usadas)
        usadas.add(cor)
        
        palette[i] =  cor

    palette[-1] = "#BCBCBC"
    return palette

def plot_clusters(X, labels, title, axis):
    if labels is not None:
        cluster_palette = gerar_palette_clusters(set(labels))
    else:
        cluster_palette = None
    reduced = PCA(n_components=2).fit_transform(X)
    sns.scatterplot(x=reduced[:,0], 
                    y=reduced[:,1], 
                    hue=labels, 
                    palette=cluster_palette, 
                    ax=axis)
    axis.set_title(title)

class TextClusterEnsemble(BaseEstimator, ClusterMixin):
    def __init__(self, n_jobs=6, plot=False, **final_params):
        self.plot = plot
        self.__it_per_model = 5
        self.n_jobs = n_jobs
        self.__labels_list = []
        self.__final_labels = []
        self.__co_matrix = None
        self.__valid_indices = None
        self.__model_params = {
            HDBSCAN: dict(min_cluster_size=[80, 50, 80, 50, 30], 
                    min_samples=([80] * 2) + ([110] * 3), 
                    metric= ['cosine'] * self.__it_per_model, 
                    cluster_selection_method=['eom'] * self.__it_per_model,
                    n_jobs=[6] * self.__it_per_model),

            DBSCAN: dict(eps=[2.4, 2.4, 2.5, 2.4, 2.5], 
                    min_samples=[50, 80, 80, 110, 140], 
                    metric=['euclidean'] * self.__it_per_model, 
                    n_jobs=[6] * self.__it_per_model),

            AgglomerativeClustering: dict(
                metric=final_params.get('metric', 'precomputed'),
                linkage=final_params.get('linkage', 'average'),
                n_clusters=final_params.get('n_clusters', 2)
            )
        }

    def has_plot(self):
        return self.plot

    def set_params(self, model, params):
        self.__model_params[model] = params

    def get_params(self, model, iteration=None):
        params = self.__model_params[model]
        if iteration is not None:
            return {p: params[p][iteration] for p in params.keys()}

        return params
    
    def get_matrix(self, normalized=False):
        mtx = self.__co_matrix
        return mtx if not normalized else mtx / mtx.max()
    
    def __create_matrix(self, size):
        cooc_matrix = np.zeros((size, size))

        for labels in self.__labels_list:
            for i in range(size):
                for j in range(size):
                    if labels[i] != -1 and labels[j] != -1 and labels[i] == labels[j]:
                        cooc_matrix[i, j] += 1

        np.fill_diagonal(cooc_matrix, 0)
        in_any_cluster = (cooc_matrix > 0).any(axis=1)
        self.__valid_indices = np.where(in_any_cluster)[0]
        self.__co_matrix = cooc_matrix[np.ix_(self.__valid_indices, self.__valid_indices)]

        self.__co_matrix /= len(self.__labels_list)


    def __run_model(self, model, X, axes=None, text=None, final=None):
        if final is None:
            label = model.fit_predict(X)
            self.__labels_list.append(label)
        else:
            label = model.fit_predict((1 - self.get_matrix(True)))
            self.__final_labels = np.full(X.shape[0], -1)
            self.__final_labels[self.__valid_indices] = label
            label = self.__final_labels

        if axes and text:
            mask = label != -1
            plot_clusters(X[mask], label[mask], text, axes)
            print(f"{text}")
            print(f"sil:{silhouette_score(X[mask], label[mask])}")
            print(f"ch:{calinski_harabasz_score(X[mask], label[mask])}")
            print(f"db:{davies_bouldin_score(X[mask], label[mask])}")

    def fit(self, X, y=None):
        text_list = [None, None, None]
        n_samples = X.shape[0]
        n_it = self.__it_per_model
        models = list(self.__model_params.keys())

        if self.plot:
            fig = plt.figure(figsize=(20, 5))
            gs = gridspec.GridSpec(2, n_it+1, width_ratios=[1, 1, 1, 1, 1, 2])
            axes = [[plt.subplot(gs[0, x]) for x in range(n_it)],
                    [plt.subplot(gs[1, x]) for x in range(n_it)]]
            axes2 = plt.subplot(gs[:, n_it])
            text_list = ["HDBSCAN", "DBSCAN"]

        for m in range(len(models)-1):
            for i in range(self.__it_per_model):
                model = models[m](**self.get_params(models[m], iteration=i))
                self.__run_model(model, X, axes[m][i], f"{text_list[m]}-{i+1}")

        self.__create_matrix(n_samples)

        model = models[-1](**self.get_params(models[-1]))
        self.__run_model(model, X, axes2, "Clustering Ensemble", final=True)

        if self.plot:
            plt.savefig(f"{PLOT_MLFLOW_ARTIFACTS}/plot_clusters-ruido.png")
            plt.show()

        return self
    
    def fit_predict(self, X):
        self.fit(X)
        return self.__final_labels
    
    def predict(self, X):
        return self.__final_labels
        

#%% JACCARD AND CLUSTER DISTANCE FUNCTIONS
def get_jaccard(df):
    jaccard_score = []
    cluster_terms = {}
    labels_set = list(df['labels'].unique())
    
    for l in labels_set:
        label_text = df[df['labels'] == l]['post']
        cluster_terms[l] = set(" ".join(label_text.astype(str).tolist()).split(" "))

    for c in list(combinations(labels_set, 2)):
        inter = cluster_terms[c[0]].intersection(cluster_terms[c[1]])
        union = cluster_terms[c[0]].union(cluster_terms[c[1]])

        jaccard_score.append(len(inter) / len(union))

    return np.mean(jaccard_score)

def get_cluster_distance(df):
    path = f"{PLOT_MLFLOW_ARTIFACTS}/cluster_distances.png"

    df_clean = df[df['labels'] != -1]
    embeddings_df = pd.DataFrame(df_clean["embedding"].tolist(), index=df_clean.index)
    embeddings_df.columns = [f"embedding_{i}" for i in range(embeddings_df.shape[1])]
    embeddings_df = pd.concat([embeddings_df, df_clean["labels"]], axis=1)

    centroids = embeddings_df.groupby("labels").mean().values
    dist_matrix = pairwise_distances(centroids)
    mean_dist = dist_matrix[np.tril_indices(len(centroids), k=-1)].mean()

    plt.figure(figsize=(8, 6))
    sns.heatmap(dist_matrix, annot=True, cmap="YlGnBu")
    plt.title("Média da Distância entre Clusters")
    plt.savefig(path)

    return mean_dist, path

#%% CLEANNING SIMILARITY
def cleaning_similarity(X):
    similarity_matrix = cosine_similarity(X)

    threshold = 0.99
    to_drop = set()

    for i in range(similarity_matrix.shape[0]):
        if i in to_drop:
            continue
        for j in range(i + 1, similarity_matrix.shape[1]):
            if similarity_matrix[i, j] > threshold:
                to_drop.add(j)
    
    return to_drop

#%% LOADING DATA
df_posts = data_loader.load_posts()
df_posts = pd.merge(
    df_posts,
    data_loader.load_posts("silver")[['SK_post','yaked']],
    on='SK_post',
    how='inner')

X  = np.vstack(df_posts["embedding"].values)
not_in_set = np.setdiff1d(np.arange(X.shape[0]), list(cleaning_similarity(X)))
X = X[not_in_set]

# %% RUN PIPELINE
pipeline = Pipeline([
    ("scaler", MinMaxScaler()),
    ("ensemble_cluster", TextClusterEnsemble(plot=True))
]) 

mlflow.set_experiment(experiment_id=EXPERIMENT_ID)
with mlflow.start_run():
    pipeline.fit(X)
    esb = pipeline.named_steps['ensemble_cluster']
    mlflow.log_params(esb.get_params(AgglomerativeClustering))

    labels = esb.predict(X)
    mask = labels != -1

    if len(set(labels[mask])) > 1:
        sil = silhouette_score(X[mask], labels[mask])
        db = davies_bouldin_score(X[mask], labels[mask])
        ch = calinski_harabasz_score(X[mask], labels[mask])

        mlflow.log_metric("silhouette_score", sil)
        mlflow.log_metric("davies_bouldin_score", db)
        mlflow.log_metric("calinski_harabasz_score", ch)

    full_labels = np.full(len(df_posts["embedding"]), -1)
    full_labels[not_in_set] = labels
    df_posts['labels'] = full_labels
    csv_path = f"{DATA_MLFLOW_ARTIFACTS}/clustering_ensemble-ruido.csv"
    df_labels = df_posts[['SK_post', 'yaked', 'labels']].rename(columns={'yaked': 'post'})
    df_labels.to_csv(csv_path, index=False)

    dist_mean, dist_path = get_cluster_distance(df_posts[['embedding', 'labels']])

    mlflow.log_metric("jaccard_mean_score", get_jaccard(df_labels))
    mlflow.log_metric("distance_mean_centroids", dist_mean)

    if esb.has_plot():
        mlflow.log_artifact(f"{PLOT_MLFLOW_ARTIFACTS}/plot_clusters-ruido.png", artifact_path="figures")

    mlflow.log_artifact(dist_path, artifact_path="figures")
    mlflow.log_artifact(csv_path, artifact_path="results")

    mlflow.set_tag("mlflow.runName", f"AGG-New-DBSCAN-3")

# %%
