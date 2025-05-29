#%%
import s3fs
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.cluster import KMeans, HDBSCAN, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.pipeline import Pipeline

from tqdm import tqdm
from itertools import combinations
from collections import defaultdict
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel
from concurrent.futures import ThreadPoolExecutor, as_completed


BUCKET_PATH = "s3://bsky-posts-lake"
GOLD_POSTS_PATH = f"{BUCKET_PATH}/gold/fato/posts"
SILVER_POSTS_PATH = f"{BUCKET_PATH}/silver/fato/posts"

PLOT_MLFLOW_ARTIFACTS = f"../Assets/Mlflow Artifacts"
DATA_MLFLOW_ARTIFACTS = f"../Data"
EXPERIMENT_ID = 793754980848381686

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
#%% MODEL CLASS
def plot_clusters(X, labels, title, axis):
    cluster_palette = {0: "#FF7468",
                       1: "#61A8E2",
                       2: "#84ED6C",
                       3: "#CC73E7",
                       4: "#E7D973",
                       5: "#737DE7",
                       6: "#552D58",
                       7: "#225030",
                       8: "#33686C",
                       -1: "#BCBCBC"}
    reduced = PCA(n_components=2).fit_transform(X)
    sns.scatterplot(x=reduced[:,0], 
                    y=reduced[:,1], 
                    hue=labels, 
                    palette=cluster_palette, 
                    ax=axis)
    axis.set_title(title)

class TextClusterEnsemble(BaseEstimator, ClusterMixin):
    def __init__(self, n_cluster, n_jobs=6, plot=False):
        self.plot = plot
        self.n_jobs = n_jobs
        self.labels_list_ = []
        self.final_labels_ = []
        self.model_params_ = dict(hdbscan = dict(min_cluster_size=80, 
                                            min_samples=80, 
                                            metric= 'cosine', 
                                            cluster_selection_method="eom",
                                            n_jobs=6),
                                dbscan = dict(eps=0.4, 
                                            min_samples=80, 
                                            metric='cosine', 
                                            n_jobs=6),
                                kmeans = dict(n_clusters=n_cluster,
                                            init='random',
                                            n_init=30,
                                            random_state=42),
                                final = dict(metric='precomputed',
                                            linkage='average',
                                            n_clusters=n_cluster))
        
    def has_plot(self):
        return self.plot

    def set_params(self, model, params):
        self.model_params_[model] = params

    def get_params(self, model):
        return self.model_params_[model]

    def _run_model(self, model, X, axes, text):
        if not isinstance(model, AgglomerativeClustering):
            label = model.fit_predict(X)
            self.labels_list_.append(label)
        else:
            label = model.fit_predict((1 - self.co_matrix_))
            self.final_labels_ = label

        if axes and text:
            plot_clusters(X, label, text, axes)
        
        

    def fit(self, X, y=None):
        text_list = [None, None, None]
        fig, axes = None, None
        n_samples = X.shape[0]
        self.co_matrix_ = np.zeros((n_samples, n_samples))

        if self.plot:
            fig = plt.figure(figsize=(20, 15))
            gs = gridspec.GridSpec(2, 3, height_ratios=[1, 2])
            axes = [plt.subplot(gs[0, x]) for x in range(3)] + [plt.subplot(gs[1, :])]
            text_list = ["HDBSCAN Clustering", "DBSCAN Clustering", "Kmeans Clustering"]

        model = HDBSCAN(**self.model_params_['hdbscan'])
        self._run_model(model, X, axes[0], text_list[0])
        model = DBSCAN(**self.model_params_['dbscan'])
        self._run_model(model, X, axes[1], text_list[1])
        model = KMeans(**self.model_params_['kmeans'])
        self._run_model(model, X, axes[2], text_list[2])

        for labels in self.labels_list_:
            for i in range(n_samples):
                for j in range(n_samples):
                    if labels[i] != -1 and labels[j] != -1 and labels[i] == labels[j]:
                        self.co_matrix_[i, j] += 1
        self.co_matrix_ /= len(self.labels_list_)

        model = AgglomerativeClustering(**self.model_params_['final'])
        self._run_model(model, X, axes[3], "Clustering Ensemble")

        if self.plot:
            plt.savefig(f"{PLOT_MLFLOW_ARTIFACTS}/plot_clusters.png")
            plt.show()

        return self
    
    def fit_predict(self, X):
        self.fit(X)
        return self.final_labels_
    
    def predict(self, X):
        return self.final_labels_
        
#%% LOADING DATA FUNCTION
def loading_table(path):
    s3 = s3fs.S3FileSystem()
    parquet_files = s3.glob(path)
    df_posts = pd.DataFrame()
    for file in parquet_files:
        with s3.open(file) as f:
            df_posts = pd.concat([df_posts, pd.read_parquet(f)])
    return df_posts

#%% COHERENCE AND JACCARD FUNCTIONS
dictionary = None
def compute_coherence_for_cluster(texts_cluster):
    if len(texts_cluster) < 2:
        return None
    
    bow_corpus = [dictionary.doc2bow(doc) for doc in texts_cluster]
    lda_model = LdaModel(
        corpus=bow_corpus,
        id2word=dictionary,
        num_topics=5,
        passes=3,
        random_state=42
    )
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=texts_cluster,
        corpus=bow_corpus,
        dictionary=dictionary,
        coherence='c_npmi'
    )
    return coherence_model.get_coherence()

def get_coherence(texts, labels):
    global dictionary
    cluster_docs = defaultdict(list)
    for doc, label in zip(texts, labels):
        cluster_docs[label].append(doc.split())

    dictionary = Dictionary([txt.split() for txt in texts])
    tasks = [docs for label, docs in cluster_docs.items() if len(docs) >= 2]

    coherences = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(compute_coherence_for_cluster, docs) for docs in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing coherence"):
            coherence = future.result()
            if coherence is not None:
                coherences.append(coherence)
    return np.mean(coherences) if coherences else 0.0

def get_jaccard(df):
    jaccard_score = []
    cluster_terms = {}
    labels_set = list(df['labels'].unique())
    
    for l in labels_set:
        label_text = df_text_labels[df_text_labels['labels'] == l]['text']
        cluster_terms[l] = set(" ".join(label_text.astype(str).tolist()).split(" "))

    for c in list(combinations(labels_set, 2)):
        inter = cluster_terms[c[0]].intersection(cluster_terms[c[1]])
        union = cluster_terms[c[0]].union(cluster_terms[c[1]])

        jaccard_score.append(len(inter) / len(union))

    return np.mean(jaccard_score)

#%% LOADING DATA

df_posts = loading_table(f"{GOLD_POSTS_PATH}/*/*.parquet")
path_silver = f"{SILVER_POSTS_PATH}/*/*.parquet"
df_posts = pd.merge(
    df_posts,
    loading_table(path_silver)[['SK_post','yaked']],
    on='SK_post',
    how='inner')

X  = np.vstack(df_posts["embedding"].values)
text_posts = df_posts['yaked'].to_list()

# %% RUN PIPELINE
pipeline = Pipeline([
    ("scaler", RobustScaler()),
    ("ensemble_cluster", TextClusterEnsemble(n_cluster=9, plot=True))
])

mlflow.set_experiment(experiment_id=EXPERIMENT_ID)
with mlflow.start_run():
   pipeline.fit(X)
   esb = pipeline.named_steps['ensemble_cluster']
   mlflow.log_params(esb.get_params('final'))

   labels = esb.predict(X)
   mask = labels != -1

   if len(set(labels[mask])) > 1:
       sil = silhouette_score(X[mask], labels[mask])
       db = davies_bouldin_score(X[mask], labels[mask])
       ch = calinski_harabasz_score(X[mask], labels[mask])

       mlflow.log_metric("silhouette_score", sil)
       mlflow.log_metric("davies_bouldin_score", db)
       mlflow.log_metric("calinski_harabasz_score", ch)

   df_text_labels = pd.DataFrame({'text': text_posts, 'labels': labels})
   df_text_labels.to_csv(f"{DATA_MLFLOW_ARTIFACTS}/clustering_ensemble.csv", index=False)

   df_text_labels = df_text_labels[df_text_labels['labels'] != -1]
   mlflow.log_metric("jaccard_mean_score", get_jaccard(df_text_labels))

   if esb.has_plot():
       mlflow.log_artifact(f"{PLOT_MLFLOW_ARTIFACTS}/plot_clusters.png", artifact_path="figures")
   mlflow.log_artifact(f"{DATA_MLFLOW_ARTIFACTS}/clustering_ensemble.csv", artifact_path="results")

#%%