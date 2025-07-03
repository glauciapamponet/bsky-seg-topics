# bsky-seg-topics

O Bluesky é uma plataforma de rede social no formato microblog que surgiu em 2021, criada como prova de conceito para apresentar a ideia de rede social descentralizada, onde os perfis de usuários estão hospedados em servidores que não são necessariamente da empresa dona da rede social. Por ter um formato de uso extremamente parecido com o antigo Twitter, o Bluesky vem atraindo usuários de diferentes nacionalidades. Em janeiro de 2025, a plataforma atingiu os 30 milhões de usuários. 

## Coleta de Dados
O bluesky disponibiliza de forma aberta uma biblioteca para python atproto, que fornece interface de cliente para a API da rede social, através do AT Protocol. Com ela é possível realizar acesso a uma conta de usuário e realizar ações de leitura e escrita dentro do perfil, além de gerenciamento de perfil.

A partir do uso dessa API é possivel coletar posts de um feed determinado pelo cliente, podendo ser o feed da conta do usuário cliente ou o feed principal com posts populares, denominado What's Hot. O limite de coleta por requisição é de 100 publicações e não existe limite de requisições.

Sendo assim, considerando o contexto temporal e as características da API, a coleta de posts foi distribuída em torno de 15 dias, com a realização da coleta de 200 posts do feed What's Hot por hora a cada dia. Essa coleta foi orquestrada via função Lambda da AWS e os posts armazenados em formato csv para cada requisição.

| Atributo   | Tipo | Descrição      |
|:------ |:----: |:------:      |
| author  | String    | Usuário autor do post  |
| text    | String    | Texto do post |
| created_at  | Timestamp    | Horário de postagem |
| like_count  | Integer    | Quantidade de likes |
| repost_count  | Integer    | Quantidade de repostagens |
| quote_count  | Integer    | Quantidade de citações |
| reply_count  | Integer    | Quantidade de respostas |

## Datalake de Posts
![Esquema de Datalake](Assets/Analysis%20Images/datalake.png)

A coleta de dados dada a partir do fornecimento da API do Bluesky foi feita entre os dias 15 e 31 de dezembro de 2024. Através de um trigger em função Lambda foi possível coletar os posts a cada hora dos 15 dias, na tentativa de amenizar duplicatas considerando os posts em maior evidência. 

Um Bucket S3 foi disponibilizado como datalake inicialmente recebendo os arquivos csv gerados pelo Lambda que em seguida passaram pela orquestração de Jobs do AWS Glue para que ocorresse a disposiçao de arquitetura Medallion, com uma rotina de carga D-1. 

**Camada Bronze:** Disponibilizada de forma particionada, teve como objetivo agregar todo o conjunto de dados em um repositório só, transicionando os dados do formato de coleta(.csv) para parquet.<br>**Camada Silver:** Disponibilizada com o conteúdo dos posts limpos e separados das informações de dimensão. O maior trabalho de limpeza e refinamento em PLN foi realizado aqui, garantindo que o conjunto pudesse ser usado para análise exploratória tardiamente. Além da limpeza de stopwrods, caracteres especiais e valores numéricos, foi realizado um refinamento usando o extrator de palavras chave não supervisionado [YAKE](https://pypi.org/project/yake/), possibilitando maior relevância no conteúdo.<br>**Camada Gold:** Disponibilizada com o conteúdo vetorizado através de um modelo Sentence Tranformer ([all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)) para receber os pipelines de Machine Learning.

## Análise Exploratória
O período de 15 de coleta de postagens do Bluesky envolvendo a API que recolhe registros do feed "What's Hot" levantou a cada hora de cada dia um montante de 100 posts via trigger de Lambda Function. A atividade totalizou uma coleta de aproximadamente 36.000 registros de posts, desconsiderando os posts que estavam em alta no momento da coleta, mas que não pertenciam a uma data no intervalo escolhido. Desse total, após remoção de duplicatas, limpeza de stopwords e extração de palavras-chave (resultando em registros vazios ou nulos onde se encontravam repetições da mesma palavra ou apenas citações de outros posts), esse número decaiu para 18.425 posts.

![Wordcloud e Frequência Top 20](Assets/Analysis%20Images/wordcloud-barplot.png)

Foi possível perceber que o período usado para a extração de posts influenciou completamente as palavras mais populares na rede daquele momento, o que era de se esperar. Existe uma evidente diferença entre a palavra mais citada no dataset, relacionada ao natal e a palavra menos citada que provavelmente provém de tópico político.

![Análise Temporal](Assets/Analysis%20Images/plot_temporal.png)

A distribuição da quantidade de posts conforme os dias do intervalo se manteve com poucas alterações mesmo após a limpeza, levando em fator a soma de posts por todos os dias do período. <br>Já a distribuição agrupada por dias da semana mostrou uma queda perceptível na sexta-feira (Friday) acompanhada de um alto range do intervalo de confiança. Observando o calendário correspondente ao período, nota-se que o espaço de 16 dias da coleta contemplou 3 vezes os dias Domingo (Sunday), Segunda-feira(Monday) e Terça-feira(Tuesday) equanto que Quarta-feira(Wednesdey), Quinta-feira(Thirsday), Sexta-feira(Friday) e Sábado(Saturday), rotacionaram uma vez a menos. O intervalo de confiança discrepante em Friday pode sugerir um recuo na criação de posts evidentes em uma das 2 datas que contemplam o dia. <br>A distribuição horária nas postagens mostrou variabilidade explicável, considerando que a coleta de horário se dá em hora local dos posts, sendo maior parte deles dentro de fusos dos Estados Unidos. O trecho com menor contagem na faixa de horario da madrugada consolida a falta de atividade de usuários por sono, além dos picos relativos e absolutos consolidarem os horarios de almoço e fim do dia como periodos de maior atividade. 

### Observando palavras de maior frequência
![Correlação e Cobertura de Palavras](Assets/Analysis%20Images/cobertura.png)

O vocabulário do conjunto de dados reuniu em todo aproximadamente 23.385 palavras citadas e adotadas como palavras chave ou de evidência para identificação. Consultando arbitrariamente as palavras mais usadas com frequencia em posts acima de 150, foi extraído um total de 97 palavras mais citadas, representando 0,4% de todo o vocabulário. A cobertura, que determina quantos posts do dataset possuem ao menos uma das 97 mais citadas apresenta valores abaixo dos 20%, o que esclarece a possivel discrepância entre a presença de tópicos mais definidos e a presença de tópicos mais individuais, além de uma alta variabilidade de vocabulário de menor frequência. <br>A rede de relações entre essas 97 palavras mais uma vez endossa a presença do feriado nas conversas da plataforma no período, uma vez que ao centro do grafo, onde vemos palavras relacionadas às festas, está a parte mais pesada de vértices. Ao redor, é possivel encontrar relações a assuntos políticos ('public' <> 'health'), artisticos ('nature' <> 'photography') e diversos ('social' <> 'media').

## Clustering
A ideia para o projeto é encontrar estratégias de segmentação que possam trazer uma apresentação de tópicos dentro dos dados coletados. Para isso, foram considerados algoritmos de clustering de dados. Nesse estudo foram escolhidos três modelos que passaram por avaliação com permutação de hiperparâmetros (Grid Search) e em seguida produziram resultados para uma matriz de coocorrência com seus melhores resultados avulsos.
Para os três algoritmos testados foram disponibilizados os dados vetorizados do DataLake e transformados de forma escalar em dois formatos: 
- **RobustScaler**: dimensionamento dos embeddings de acordo com seu IQR (intervalo inter-quartil). Útil em padronizar dados com outliers ou ruídos.
- **MinMaxScaler**: dimensionamento dos embeddings no intervalo [0,1] de forma padrão, usando valores maximo e mínimo.

A seleção do conjunto de dados também passou por uma limpeza de valores duplicados visando uma melhor mineração de ruídos. Foram identificados alguns textos identicos de mesmos autores, provinentes de páginas de notícias que acabavam poluindo a distribuição. Sendo assim, um corte baseado na distância via similaridade de cosseno de até **0.99** foi realizado a fim de solucionar essa questão. 

Entre os algoritmos avaliados estão **KMeans**, **DBSCAN** e **HDBSCAN**.

Usando o Mlflow, inicialmente foi realizada uma avaliação para cada tipo de modelo através de **Grid Search**, contando como estimativa de alguns paramêtros a observação do comportamento dos dados aos modelos, como no caso do Kmeans, onde foi utilizado o Elbow Method (Método do Cotovelo) para determinar em quais valores de K a inércia do modelo sofreria alteração. O método também foi usado para estipular valores para DBSCAN observando o comportamento do valor de epsilon (raio de área mínima para formação de um cluster) em relação aos dados.<br>Ao todo, foram feitas mais de 300 execuções nos experimentos destinados a cada tipo de modelo para definir os melhores parametros para o Clustering Ensemble. Nos experimentos do Mlflow, foram registrados os hiperparâmetros de cada execução, bem como as métricas de avaliação básicas: **silhoute, davies bouldin e calinski harabasz**.

O pipeline adotado para o esquema de Clustering Ensemble contou com 10 camadas internas e uma camada final de saída das labels. Após execução de experimentos, foram escolhidas 5 variações do algoritmo DBSCAN e outras 5 variações do algoritmo HDBSCAN, por serem os mais bem sucedidos entre os 3 tipos de algoritmo explorados. Essas camadas produziram seus respectivos agrupamentos e os resultados foram sumarizados em uma matriz de coocorrência, que determina quais dados foram agrupados pelos 10 algoritmos no mesmo rótulo.<br> Uma vez que o modelo é constituido de algoritmos que produzem ruído, a matriz de coocorrência desconsidera conteúdos que não foram inseridos em nenhum agrupamento durante as 10 execuções de clustering, caracterizando ruídos absolutos. Ela é submetida a um novo processo através do algoritmo **AglomerativeClustering**, produzindo assim a rotulação final para o conjunto de dados.

## Execução

A execução das camadas do pipeline de Ensemble não mostrou grande variabilidade em resultados. Todas as 10 camadas produziram a mesma quantidade de grupos, sendo assim possível determinar com clareza a separação arbitrária da camada final em K=2.

![Métricas de Camadas Internas Ensemble](Assets/Analysis%20Images/comparação-layers.png)

É possível perceber que, no primeiro gráfico, mesmo para os melhores parâmetros, o conjunto de dados não mostra um desempenho ótimo em termos de separação dos grupos do ponto de vista das fronteiras de decisão que compoem esses clusters. Também fica claro que apesar da baixa variabilidade da avaliação em cada camada, o algoritmo HDBSCAN apresentou clusters menos propensos a sobreposição, apesar de ainda próximos.<br>
A avaliação acerca da variabilidade de dados intra e entre clusters no segundo gráfico mostra que no geral o conjunto teve uma variabilidade com amplitude baixa, sendo essa próxima a zero nas camadas HDBSCAN. Já em DBSCAN, com uma amplitude baixa, porem com desempenho crescente, a avaliação Callinski-Harabasz mostra potencial de maior distinção entre clusters, caso prosseguisse como ajuste de parâmetros.<br>
Já a avaliação de similaridade geral no terceiro gráfico indica melhor resultado nos modelos HDBSCAN que conclui resultados consistentes dentro das 5 camadas, mostrando as menores dispersões intra-clusters das 10 iterações. Já em DBSCAN, houve certa oscilação entre as iterações, tendo uma geral mediana.

![Distancia entre Clusters e Métricas Finais do Ensemble](Assets/Analysis%20Images/metricas-finais-heatmap.png)

Na camada final do Ensemble é possivel observar que as métricas de avaliação interna e externa decaíram quando comparadas às das camadas internas do HDBSCAN, porém muito próximas às de DBSCAN.<br>
Isso pode indicar que os registros que foram agrupados em DBSCAN mas não em HDBSCAN oferecem impacto o suficiente para que o clustering hierárquico tenda a uma performance mais próxima a DBSCAN. Uma evidência seria a comparação das métricas finais com as de DBSCAN, em que a variabilidade de dados sinaliza ser mais alta em comparação com as iterações HDBSCAN, enquanto os limites de clusters são menos marcados e a similaridade é menor do que em HDBSCAN.<br>
A distância entre centróides dos dois clusters não conclui que estão em locais de completos opostos no espaço vetorial, mas é significativa o suficiente para permitir que exista distância distinguível entre os grupos.

## Resultados

![Wordcloud dos Clusters Gerados](Assets/Analysis%20Images/wordcloud-clusters.png)

Com os resultados dos clusters gerados é possível observar que existe distinção semantica bem demarcada entre o conjunto de palavras. Enquanto no cluster 1, fica evidente a carga generalista com enfase nos assuntos voltados ao feriado, mesmo sendo possivel observar tópicos que possivelmente gerariam subclusters, o cluster 2 carrrega caráter mais político e de atualidades, com foco nos tópicos que foram notícia dentro do período. A discrepância da variabilidade de palavras torna o cluster 2 mais definido em termos de semântica e assunto do que o cluster 1.

A popularidade dos posts no dataset é analisada a partir dos parametros de interação, onde é possivel catalogar a interação ativa, onde os posts recebem comentários (Quotes e Replies) e a interação passiva, onde apenas uma reação simples é registrada (Likes e Reposts).
Quando observada a popularidade (os índices de recepção e interação com os posts) dos tópicos de cada cluster, é possivel notar que, apesar da diferença na variabilidade de tópicos, o cluster 2 obteve mais interações do que o cluster 1.

![Distribuição da Interação para cada Cluster](Assets/Analysis%20Images/boxplot-clusters.png)

Os formatos de distribuição das interações (Likes, Reposts, Quotes e Respostas) de cada cluster mostram semelhanças, indicando a desproporção de popularidade para ambos os grupos(poucos posts alcançaram picos de popularidade, mesmo desconsiderando outliers máximos), mas ainda evidenciando que o cluster 2 obteve maior popularidade em seus assuntos do que o cluster 1.

Uma vez que a distribuição de interações nos posts apresenta acentuada assimilaridade, olhando então do ponto de vista do valor mediano de interações, é possível observar o comportamento temporal de interações e recepção dos grupos a cada período de hora.

![Interações a Cada Hora](Assets/Analysis%20Images/interacoes-hora.png)

Aqui fica evidente que os tópicos do cluster 2 apresenta maior oscilação de interação, além de carregar uma média geral para cada tipo de interação levemente maior que a dos tópicos no cluster 1. Em relação a horário notáveis, o cluster um teve maior estabilidade nas quantidades medianas de reação enquanto o cluster 2 apresentou picos de atividade na madrugada e no início da tarde, encontrados em todos os tipos de interações.

No contexto dos tópicos mais citados dentro de cada agrupamento, o resultado dos 5 tópicos de cada cluster mostra que ambos tem muito em torno do mesmo assunto. Replicando a estrutura do dataset geral, as palavras mais pontuadas representam pouco em relação a todo o vocabulário encontrado em cada um dos grupos, principalmente no Cluster 1, em que o carater generalista é mais forte, a quantidade de posts é maior e apresenta variabilidade de assuntos. 

![Tópicos Mais Citados](Assets/Analysis%20Images/cobertura-topicos.png)

Em relação aos parâmetros de interação, fica claro a diferença escalar entre a interação passiva e a interação ativa, evidenciando a alta em Likes. No cluster 2 a interação ativa apresenta aumento consideravel em comparação ao cluster 1, principalmente em Quotes. Em geral, dentro de suas proporções as quatro métricas apresentam pouca variação para os tópicos, reforçando a ideia da participação deles nos mesmos assuntos.

Analisando os dados em relação à autoralidade dos posts, fica evidente uma separação de estilo de conteúdo que segue a linha temática dos clusters.

![Presença de Autores](Assets/Analysis%20Images/presença-top-10.png)

A lista de contas mais ativas dentro do universo dos clusters segmentados mostra que o cluster 2, por seu caráter político e de atualidades, apresenta autores focados em informação. Já o gênero de conteúdo dos autores mais ativos no cluster 1 espelha a categoria de variedades, sendo essas contas pessoais ou de conteúdo de entretenimento. A presença de posts em cada cluster para o conjunto de autores mostra também que, apesar de produzir conteúdo direcionado, os autores dos posts no cluster 2 podem produzir conteúdo generalista e vice-versa, sendo para esse cluster uma proporção maior do que para o cluster 1.

Pela ótica da interação mais expressiva, a interação passiva de Likes, os autores encontrados em ambos os clusters não integram os autores mais ativos, tendo contabilizado poucos posts no agrupamento dentro do período.

![Tópicos e Likes - C1](Assets/Analysis%20Images/top-author-0.png)

Os 5 autores mais populares em recepção de Likes dentro do grupo 1 compartilham vocabulário relacionado ao feriado, com esceção do segundo mais popular, que apresenta conteúdo generalista. A presença de contas de celebridades no conjunto ajuda a entender a alta popularidade nos posts.

![Tópicos e Likes - C2](Assets/Analysis%20Images/top-author-1.png)
O grupo 2 reforça o caráter informativo quando observado o vocabulário dos autores mais populares. Há a presença de autor de intersecção (autores que postaram em tópicos de ambos os grupos). Como no grupo 1, aqui é perceptivel a enfase em um assunto de evidência no período da coleta.

## Considerações
Apesar de bem demarcado, o processo de coleta, agrupamento e exploração apresentou desafios que podem evidenciar melhoras em desenvolvimentos futuros:<br>

- **Segmentação de Coleta:** por apresentar alta variabilidade de assuntos dentro do Feed fornecido pela API do Bluesky, o conjunto de dados pode apresentar extrema distinção semântica, exigindo maior tratamento na vetorização ou dificultando o clustering pelo diagnóstico de muitos ruidos. Isso pode ser um ponto de interferência, principalmente quando se considera marcos temporais, como no caso desse projeto (período de festas). Desse modo, a segmentação por assunto chave pode direcionar a busca e produzir um conjunto de maior qualidade para segmentação.
- **Processamento de Linguagem Natural:** a extração de conteúdo chave é oportuno dentro da separação de tokens no conjunto de texto no momento da limpeza. Entretanto, é importante explorar o limiar de importância das palavras chave ao extrair, ou corre-se o risco de prejudicar a semântica do texto.
- **Visualização de Grupos:** sendo um dado não estruturado que apresenta centenas de dimensões, os embeddings limitam a visualização de dados. A redução de dimensionalidade pode não ser uma opção pela perda de explicabilidade dos dados no prodesso de redução, o que obriga a formas de "visualização" alternativas do conjunto de grupos.
- **Plataforma de Orquestração:** o uso de ambiente para orquestrar o pipeline de coleta deve ser escolhido com cuidado e o uso consciente, se depender de custos financeiros.



