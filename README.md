# bsky-seg-topics

O Bluesky é uma plataforma de rede social no formato microblog que surgiu em 2021, criada como prova de conceito para apresentar a ideia de rede social descentralizada, onde os perfis de usuários estão hospedados em servidores que não são necessariamente da empresa dona da rede social. Por ter um formato de uso extremamente parecido com o antigo Twitter, o Bluesky vem atraindo usuários de diferentes nacionalidades. Em janeiro de 2025, a plataforma atingiu os 30 milhões de usuários. O possível crescimento acaba marcando abertura a exploração de público dentro da rede social através de produtores de conteúdo, marcas e publicitários que, mesmo com a recusa do Bluesky na inserção de anúncios nativos, podem investir em comunicação e engajamento o público da rede.

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
![alt text](Assets/datalake.png)

A coleta de dados dada a partir do fornecimento da API do Bluesky foi feita entre os dias 15 e 31 de dezembro de 2024. Através de um trigger em função Lambda foi possível coletar os posts a cada hora dos 15 dias, na tentativa de amenizar duplicatas considerando os posts em maior evidência. 

Um Bucket S3 foi disponibilizado como datalake inicialmente recebendo os arquivos csv gerados pelo Lambda que em seguida passaram pela orquestração de Jobs do AWS Glue para que ocorresse a disposiçao de arquitetura Medallion, com uma rotina de carga D-1. 

**Camada Bronze:** Disponibilizada de forma particionada, teve como objetivo agregar todo o conjunto de dados em um repositório só, transicionando os dados do formato de coleta(.csv) para parquet.
**Camada Silver:** Disponibilizada com o conteúdo dos posts limpos e separados das informações de dimensão. O maior trabalho de limpeza e refinamento em PLN foi realizado aqui, garantindo que o conjunto pudesse ser usado para análise exploratória tardiamente. Além da limpeza de stopwrods, caracteres especiais e valores numéricos, foi realizado um refinamento usando o extrator de palavras chave não supervisionado ![YAKE](https://pypi.org/project/yake/), possibilitando maior relevância no conteúdo.
**Camada Gold:** Disponibilizada com o conteúdo vetorizado através de um modelo Sentence Tranformer (all-MiniLM-L6-v2) para receber os pipelines de Machine Learning.

## Análise Exploratória

## Pipelines de ML

## Análise de Resultados
