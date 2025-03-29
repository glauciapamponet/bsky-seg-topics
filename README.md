# bsky-seg-topics

O Bluesky é uma plataforma de rede social no formato microblog que surgiu em 2021, criada como prova de conceito para apresentar a ideia de rede social descentralizada, onde os perfis de usuários estão hospedados em servidores que não são necessariamente da empresa dona da rede social. Por ter um formato de uso extremamente parecido com o antigo Twitter, o Bluesky vem atraindo usuários de diferentes nacionalidades. Em janeiro de 2025, a plataforma atingiu os 30 milhões de usuários. O possível crescimento acaba marcando abertura a exploração de público dentro da rede social através de produtores de conteúdo, marcas e publicitários que, mesmo com a recusa do Bluesky na inserção de anúncios nativos, podem investir em engajar o público da rede.

Dentro desse contexto, é interessante observar como se comporta o fluxo de maior evidência dentro da plataforma, ato que pode trazer uma analise exploratória das tendencias a se apostar como alvo de publicidade uma vez que o crescimento de Bluesky se mostra constante. Os meses finais de 2024 se mostrou um periodo de grande evidência e atividade dentro da rede, período em que foi possível constatar uma estabilização de usuários, após a volta do antigo Twitter em outubro do mesmo ano. Dentro de um contexto publicitário, os meses finais do ano fazem parte de um período crucial, devido às festas e férias escolares ou universitárias. Sendo assim, a observação de posts em evidencia dentro da plataforma nesse período é uma aposta consistente no objetivo adotado.

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
![alt text](Assets\datalake.png)

A coleta de dados dada a partir do fornecimento da API do Bluesky foi feita entre os dias 15 e 31 de dezembro de 2024. Através de um trigger em função Lambda foi possível coletar os posts a cada hora dos 15 dias, na tentativa de amenizar duplicatas considerando os posts em maior evidência. 

Um Bucket S3 foi disponibilizado como datalake inicialmente recebendo os arquivos csv gerados pelo Lambda que em seguida passaram pela orquestração de Jobs do AWS Glue para que ocorresse a disposiçao de arquitetura Medallion, com uma rotina de carga D-1. 

A camada Bronze foi disponibilizada de forma particionada, enquanto a camada Silver foi disponibilizada com o conteúdo dos posts limpos e separados das informações de dimensão. A camada Gold foi disponibilizada com o conteúdo vetorizado através de um modelo Sentence Tranformer (all-MiniLM-L6-v2) para receber os pipelines de Machine Learning.

