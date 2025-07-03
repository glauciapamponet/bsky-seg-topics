#%%
import squarify
import pandas as pd
import seaborn as sn
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import combinations
from wordcloud import WordCloud

from Assets.Code import LoadingData

from collections import Counter

data_loader = LoadingData.LoadingData()

#%%

def plot_wordcloud(data, title, axes=None):
    all_text = " ".join(data.astype(str).tolist())

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        collocations=False,
    ).generate(all_text)

    axes = plt if not axes else axes

    axes.imshow(wordcloud, interpolation='bilinear')
    axes.axis("off")
    axes.set_title(title)

#%%
df_posts = data_loader.load_posts("silver")
df_author = data_loader.load_posts("silver")
df_time = data_loader.load_posts("silver")

#%%
df_join = pd.merge(
    pd.merge(df_posts, df_author, 'inner', 'SK_author'), 
    df_time, 
    'inner', 
    'SK_time')

#%%
# Análise Geral
fig, axes = plt.subplots(1, 2, figsize=(20, 5))
plot_wordcloud(df_posts['yaked'], "WordCloud Geral", axes[0])

df_split = df_posts.copy()
df_split['splitted'] = df_split['yaked'].apply(lambda x: x.split(" "))
count = Counter([word for tokens in df_split['splitted'] for word in tokens])
top_words = pd.DataFrame(count.most_common(20), columns=['palavra', 'frequencia'])

sn.barplot(data=top_words, x='palavra', y='frequencia', ax=axes[1], color='salmon')
axes[1].set_title("Top 20 Palavras")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90)
plt.show()


#%%
# Análise Temporal de Posts
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_count = pd.DataFrame(df_join.groupby(['DIA'])['SK_post'].count()).reset_index()
day_count = pd.merge(day_count, df_time[['DIA', 'DIA_SEMANA']].drop_duplicates(), 'inner', 'DIA')
day_count['DIA_SEMANA'] = pd.Categorical(day_count['DIA_SEMANA'], categories=day_names, ordered=True)

hour_count = pd.DataFrame(df_join.groupby(['DIA', 'HORA'])['SK_post'].count()).reset_index()

def plot_temporal(df, param, axes, color, label_list, barplot=False):
    if barplot:
        sn.barplot(data=df, x=param, y='SK_post', ax=axes, color=color)
    else:
        sn.lineplot(data=df, x=param, y='SK_post', ax=axes, color=color)
    axes.set_xticks(df[param].unique())
    axes.set_title(label_list[0])
    axes.set_xlabel(label_list[1])
    axes.set_ylabel(label_list[2])

fig, axes = plt.subplots(1, 3, figsize=(25, 5))

labels = ["Posts por dia de Coleta (14/12 a 31/12)", "Dias", "Quantidade de Posts"]
plot_temporal(day_count, 'DIA', axes[0], 'skyblue', labels)
labels = ["Posts por Dia da Semana", "Dia da Semana", "Quantidade de Posts"]
plot_temporal(day_count, 'DIA_SEMANA', axes[1], 'lightgreen', labels, barplot=True)
labels = ["Posts por Hora de Postagem", "Hora de Postagem", "Quantidade de Posts"]
plot_temporal(hour_count, 'HORA', axes[2], 'salmon', labels)

plt.show()

#%%
# Analise da Correlação de tópicos e Cobertura de Posts

df_split = df_posts.copy()
df_split['splitted'] = df_split['yaked'].apply(lambda x: x.split(" "))

def data_coorelation(df):
    count = Counter([word for tokens in df['splitted'] for word in tokens])
    df_words = pd.DataFrame(count.items(), columns=['palavra', 'frequencia'])
    set_words = set(df_words[df_words['frequencia'] >= 150]['palavra'])

    top_words = count.most_common(len(set_words))
    covered_words = len(set_words)
    total_words = sum(count.values())
    sizes = [covered_words, total_words - covered_words]

    tokens = [token for sublist in df['splitted'] for token in sublist]
    top_words = dict(top_words).keys()
    window_size = 2
    G = nx.Graph()
    edges = []

    for i in range(len(tokens) - window_size + 1):
        window = tokens[i:i + window_size]
        pairs = combinations(window, 2)
        edges.extend(p for p in pairs if p[0] in top_words and p[1] in top_words and p[0] != p[1])

    for (w1, w2), weight in Counter(edges).items():
        G.add_edge(w1, w2, weight=weight)

    return sizes, G

sizes, G = data_coorelation(df_split)

fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 7]) 

ax0 = plt.subplot(gs[0])
squarify.plot(sizes, label=['Cobertas', 'Não cobertas'], color=['#66c2a5', '#fc8d62'], ax=ax0)
ax0.set_xlim(0, 100)
ax0.invert_yaxis()
ax0.set_title('Cobertura de Vocabulário\nFrequência > 150')
ax0.set_xticks([])

pos = nx.spring_layout(G, k=0.5)
edges = G.edges()
weights = [G[u][v]['weight'] for u,v in edges]

ax1 = plt.subplot(gs[1])
nx.draw_networkx_nodes(G, pos, node_size=1200, node_color='#66c2a5', ax=ax1)
nx.draw_networkx_edges(G, pos, edgelist=edges, width=[w/2 for w in weights], alpha=0.6, ax=ax1)
nx.draw_networkx_labels(G, pos, font_size=12, ax=ax1)
ax1.set_title("Grafo de Correlação - Frequencia > 150")
ax1.axis('off')

plt.tight_layout()
plt.show()
