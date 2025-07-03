#%%
import gc
import squarify
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from Assets.Code.RadarFactory import radar_factory
from Assets.Code import LoadingData

from PIL import Image
from wordcloud import WordCloud
from matplotlib_venn import venn2

from collections import Counter

data_loader = LoadingData.LoadingData()

#%% LOADING DATA
columns = ["SK_post", "SK_time", "SK_author"]
clt_esb = pd.read_csv('Data\clustering_ensemble-ruido.csv')
df_esb = pd.merge(
    clt_esb[clt_esb['labels'] != -1],
    data_loader.load_posts(),
    "inner",
    "SK_post")

del clt_esb
gc.collect()

#%% MÉTRICAS DAS LAYERS DO ENSEMBLE
metrics = ['Name', 'silhouette_avg', 'calinski_harabasz', 'davies_bouldin']
df_layers = pd.concat(
    [pd.read_csv("..\Data\HDBSCAN-runs.csv")[metrics],
     pd.read_csv("..\Data\DBSCAN-runs.csv")[metrics]]
)

df_layers['Model'] = df_layers['Name'].apply(lambda x: x.split('-')[0])
df_layers['Layer'] = [i+1 for i in range(5)] + [i+1 for i in range(5)]
df_layers.drop(columns=['Name'], inplace=True)

fig = plt.figure(figsize=(15, 6))
gs = gridspec.GridSpec(2, 3)
axes = [plt.subplot(gs[0, c]) for c in range(3)]

for i in range(3):
    sns.lineplot(data=df_layers, 
                 x="Layer", 
                 y=metrics[i+1], 
                 hue="Model", 
                 ax=axes[i])
    handles, labels = axes[i].get_legend_handles_labels()
    axes[i].xaxis.set_major_locator(ticker.MultipleLocator(1))
    axes[i].set_title(metrics[i+1])
    axes[i].legend_.remove()

fig.legend(handles, labels, loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, 1.05))
plt.tight_layout()
plt.savefig("../Assets/Analysis Images/comparação-layers.png")

#%% METRICAS LAYER FINAL ENSEMBLE
data = [
    ["calinski_harabasz_score", 172.15],
    ["davies_bouldin_score", 2.18],
    ["distance_mean_centroids", 0.72],
    ["jaccard_mean_score", 0.11],
    ["silhouette_score", 0.12]
]

fig, ax = plt.subplots(figsize=(4, 2), dpi=100)  
ax.axis('off') 

table = ax.table(
    cellText=data,
    colLabels=["Metric", "Value"],
    loc='center',
    cellLoc='center',
    colLoc='center',
    colWidths=[0.5, 0.2],
    bbox=[0.1, 0.1, 0.8, 0.8]
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

ax.set_title("Métricas Finais do Ensemble")

plt.tight_layout()
plt.savefig("../Assets/Analysis Images/metricas-finais-heatmap.png")


# %% WORDCLOUD 
def plot_wordcloud(data, title, axes=None):
    mask_coud = np.array(Image.open("../Assets/Analysis Images/wordcloud-mask.jpg"))
    all_text = " ".join(data.astype(str).tolist())

    wordcloud = WordCloud(
        width=800,
        height=400,
        mask=mask_coud,
        background_color='white',
        collocations=False,
    ).generate(all_text)

    axes = plt if not axes else axes
    axes.imshow(wordcloud, interpolation='bilinear')
    axes.axis("off")
    axes.set_title(title)

fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 2)
clusters = df_esb['labels'].unique().shape[0]
axis = [plt.subplot(gs[ax]) for ax in range(clusters)]
for c in range(clusters):
    texts = df_esb[df_esb['labels'] == c]['post']
    plot_wordcloud(texts, f"Cluster {c+1}", axis[c])

plt.subplots_adjust(wspace=-0.6, hspace=0)
plt.savefig("../Assets/Analysis Images/wordcloud-clusters.png")

#%% SPLIT DF
df_split = df_esb.copy().drop(columns=['embedding'])
df_split['splitted'] = df_split['post'].apply(lambda x: x.split(" "))
df_split.drop(columns=['post'], inplace=True)

df_split

#%% COBERTURA DE FREQUENCIA E TOPICOS POPULARES
def data_plot(df, n_words):
    count = Counter([word for tokens in df['splitted'] for word in tokens])
    top_words = dict(count.most_common(n_words))
    covered_words = sum(top_words.values())
    total_words = sum(count.values())
    sizes = [covered_words, total_words - covered_words]

    cols = ['like_count', 'repost_count', 'quote_count', 'reply_count']
    df_words = df[cols].copy()
    mean_words = dict()
    for w in top_words.keys():
        df_words[w] = df['splitted'].map(lambda x: int(w in list(x)))
        filtered = df_words[df_words[w] == 1].drop(columns=[w])
        mean_words[w] = list(dict(filtered.median()).values())
        df_words.drop(columns=[w], inplace=True)

    cols = [c.replace('count', 'median') for c in cols]
    avg_stats = pd.DataFrame(data=mean_words, index=cols)

    return sizes, avg_stats

fig = plt.figure(figsize=(12, 10))
gs = gridspec.GridSpec(2, 3, width_ratios=[0.3, 1, 1], hspace=0.35, wspace=0.1, figure=fig) 
color_cover = ['#66c2a5', '#fc8d62']
color_radar = ['b', 'r', 'g', 'm']
theta = radar_factory(5, frame='polygon')
handles = []

for c in range(0, 3, 2):
    mask = df_split['labels'] == int(c/2)
    cover, avg_words =  data_plot(df_split[mask], 5)

    ax = fig.add_subplot(gs[int(c/2), 0])
    squarify.plot(cover, label=['Cobertos', 'Não cobertos'], color=color_cover, ax=ax)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.set_title('Cobertura de Posts\nTop 5 Palavras', fontdict={'fontsize': 10})
    ax.set_xticks([])
    
    def plot_radars(col, indexes, colors):
        ax = fig.add_subplot(gs[int(c/2), col], projection='radar')
        for d, color in zip(indexes, colors):
            ax.plot(theta, avg_words.loc[d], color=color)
            ax.fill(theta, avg_words.loc[d], facecolor=color, alpha=0.25, label='_nolegend_')
        degrees = np.degrees(np.linspace(0, 2*np.pi, len(avg_words.columns), endpoint=False))
        ax.set_thetagrids(degrees, list(avg_words.columns))
        ax.legend(indexes, loc=(0.32, -0.05), labelspacing=0.3, fontsize='small')

    plot_radars(1, list(avg_words.index)[:2], color_radar[:2])
    plot_radars(2, list(avg_words.index)[2:], color_radar[2:])

fig.text(0.5, 0.93, 'Cluster 1', fontsize=10, fontweight='bold', va='center')
fig.text(0.5, 0.50, 'Cluster 2', fontsize=10, fontweight='bold', va='center')

plt.savefig('../Assets/Analysis Images/cobertura-topicos.png')

#%% Presença de autor em cada cluster (diagrama de venn) e autores mais ativos
df_esb_author = pd.merge(df_esb, data_loader.load_author(), 'inner', 'SK_author')
df_esb_author['author'] = df_esb_author['author'].map(lambda x: x.split('.')[0])
cols = [c for c in df_esb_author.columns if c not in ['embedding', 'post']]
mask = df_esb_author['labels']

df_author = [df_esb_author[cols][mask == 0], df_esb_author[cols][mask == 1]]
set_0 = set(df_author[0]['author'].unique())
set_1 = set(df_author[1]['author'].unique())
inner_set = set_0.intersection(set_1)

qtd_posts_0 = len(df_esb_author[cols][mask == 0])
qtd_posts_1 = len(df_esb_author[cols][mask == 1])

fig = plt.figure(figsize=(15, 6))
gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[2, 1], wspace=0.15, hspace=0.5)
boxcolors = ["#ffff99", "#9999ff"]
ax_big = fig.add_subplot(gs[:, 0])

v = venn2(subsets = (len(set_0)-len(inner_set), len(set_1)-len(inner_set), len(inner_set)), 
      set_labels = ('Cluster 1', 'Cluster 2'), 
      set_colors= ('yellow', 'blue'),
      ax=ax_big)
v.set_labels[0].set_fontsize(10)
v.set_labels[1].set_fontsize(10)
ax_big.set_title('Presença de autores nos Clusters')


for c in range(2):
    ax = fig.add_subplot(gs[c, 1])
    data = df_author[c].groupby('author').count().reset_index()
    data = data.sort_values(by='SK_post', ascending=False).iloc[:10]
    sns.barplot(data=data, 
                    y='author', 
                    x='SK_post', 
                    ax=ax, 
                    color=boxcolors[c])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel('Quantidade de Posts')
    ax.set_title(f'10 Autores Mais Ativos - Cluster {c+1}', fontsize=10)

plt.savefig('Assets/Analysis Images/presença-top-10.png')

#%% LIKES E AUTORES
def plot_authors_likes(cluster):
    def get_author_texts(df, auth):
        mask = df['author'] == auth
        return " ".join(df[mask]['post'].astype(str).tolist())

    top_authors = df_author[cluster].groupby('author').mean()[['like_count']]
    top_authors.sort_values(by='like_count', ascending=False, inplace=True)

    mask = df_esb_author['labels'] == cluster
    autores = list(top_authors.iloc[:5].index)
    likes = list(top_authors.iloc[:5]['like_count'])
    texts = [get_author_texts(df_esb_author[mask], aut) for aut in autores]

    n, y_pos = len(autores), np.arange(len(autores))
    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(2, n, figure=fig, height_ratios=[0.7, 1], wspace=0.15, hspace=0.1)
    plot_title = f"Top 5 Médias de Likes: Autores e Seus Tópicos - Cluster {cluster+1}"

    ax_bars = fig.add_subplot(gs[0, :])
    ax_bars.barh(y_pos, likes, color=boxcolors[cluster])
    ax_bars.set_yticks(y_pos)
    ax_bars.set_yticklabels(autores)
    ax_bars.invert_yaxis()
    ax_bars.set_xlabel("Likes")
    ax_bars.set_title(plot_title, fontdict={'size': 10})
    ax_bars.set_xlim(0, max(likes)*1.1)

    for i in range(n):
        ax_wc = fig.add_subplot(gs[1, i])
        wc = WordCloud(width=400, height=200, background_color="white").generate(texts[i])
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        ax_wc.set_title(f"{autores[i]}", fontsize=10)

    plt.tight_layout()
    plt.savefig(f"Assets/Analysis Images/top-author-{cluster}.png")

plot_authors_likes(0)
plot_authors_likes(1)

#%% DISTRIBUIÇÃO INTERACAO CLUSTERS - BOXPLOT
cols = ['like_count', 'repost_count', 'quote_count', 'reply_count']
linecolors = ['green', 'blue', 'red', 'yellow']
boxcolors = ["#BAF4BEFF", "#8BC3FFFF", "#F4BABAFF", "#F6FFAEFF"]
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))
plt.subplots_adjust(hspace=0.5, wspace=0.5)
axes = axes.flatten()

for m in range(len(cols)):
    data = df_esb[['labels', cols[m]]].rename(columns={'labels': 'Cluster'})
    data['Cluster'] = data['Cluster'] + 1
    sns.boxplot(data=data, 
                x='Cluster', 
                y=cols[m], 
                ax=axes[m], 
                color=boxcolors[m], 
                linecolor=linecolors[m], 
                showfliers=False)
    axes[m].set_xlabel('')
    
fig.supxlabel('Cluster', fontsize=10)
fig.text(0.4, .95, 'Distribuição de Interações por Cluster', fontsize=10, va='center')
plt.savefig('../Assets/Analysis Images/boxplot-clusters.png')

#%% ANALISE TEMPORAL - LIKES, REPOSTS, QUOTES e REPLIES A CADA HORA
cols = ['labels','like_count','repost_count','quote_count','reply_count','HORA']
df_esb_time = pd.merge(df_esb, 
                       data_loader.load_time()[['SK_time', 'HORA']],
                       'inner',
                       'SK_time')[cols]

group = ['labels', 'HORA']
clusters_time = [df_esb_time.groupby(group).median().loc[c].reset_index() for c in range(2)]

fig, axes = plt.subplots(2, 2, figsize=(16, 6))
plt.subplots_adjust(hspace=0.3, wspace=0.15, top=0.85, bottom=0.09)
axes = axes.flatten()

for i in range(len(cols[1:-1])):
    sns.lineplot(data=clusters_time[0], y=cols[i+1], x="HORA", ax=axes[i], label="Cluster 1")
    sns.lineplot(data=clusters_time[1], y=cols[i+1], x="HORA", ax=axes[i], label="Cluster 2")
    handles, labels = axes[i].get_legend_handles_labels()
    axes[i].set_xticks(clusters_time[0]["HORA"].unique())
    axes[i].set_ylabel(cols[i+1].replace('count', 'median'))
    axes[i].set_xlabel("Hora")
    axes[i].legend_.remove()

fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.05))
fig.text(.425, .90, 'Mediana de Interações a Cada Hora', fontsize=11, va='center')
plt.savefig("../Assets/Analysis Images/interacoes-hora.png")

#%% TESTE RADAR CHART

data = [['Sulfate', 'Nitrate', 'EC', 'OC1', 'OC2'],
        ('Basecase', [
            [0.88, 0.01, 0.03, 0.03, 0.00],
            [0.07, 0.95, 0.04, 0.05, 0.00],
            [0.01, 0.02, 0.85, 0.19, 0.05],
            [0.02, 0.01, 0.07, 0.01, 0.21],
            [0.01, 0.01, 0.02, 0.71, 0.74]])]

N = len(data[0])
theta = radar_factory(N, frame='polygon')

spoke_labels = data.pop(0)
title, case_data = data[0]
labels = ('Factor 1', 'Factor 2', 'Factor 3', 'Factor 4', 'Factor 5')

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='radar'))

fig.subplots_adjust(top=0.85, bottom=0.05)

ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
ax.legend(labels, loc=(0.9, .95), labelspacing=0.1, fontsize='small')
ax.set_title(title,  position=(0.5, 1.1), ha='center')

for d in case_data:
    line = ax.plot(theta, d)
    ax.fill(theta, d,  alpha=0.25)
ax.set_varlabels(spoke_labels)

plt.show()

#%% MUDANDO DIRETORIO
import os

os.chdir('C:/Users/glaup/Desktop/DADOS/Projetos Portifolio/Bluesky Segmentation/bsky-seg-topics')


#%%
#%%
df_author[0]