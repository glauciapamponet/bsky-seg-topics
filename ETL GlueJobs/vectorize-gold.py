print("start session")
from pyspark.sql.types import *
from datetime import timedelta
from pyspark.sql.functions import col, min, max, udf, lit
from sentence_transformers import SentenceTransformer

import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
  
sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)

gold_path = "s3://bsky-posts-lake/gold"
silver_path = "s3://bsky-posts-lake/silver"

time_path_gold = f"{gold_path}/dim/time/"
time_path_silver = f"{silver_path}/dim/time/"

author_path_silver = f"{silver_path}/dim/author/"

posts_silver = f"{silver_path}/fato/posts/"
posts_gold = f"{gold_path}/fato/posts"

model = SentenceTransformer('all-MiniLM-L6-v2')

def encode(text):
    embed = model.encode(text, convert_to_numpy=True)
    return embed.flatten().tolist()

def get_data_gold(schema, path, atribute):
    try: 
        df_gold = spark.read.format("parquet") \
            .option("header", "true") \
            .schema(schema) \
            .load(path)
        if "/dim/time/" not in path:
            path_time = "s3://bsky-posts-lake/gold/dim/time/"
            df_gold = df_gold.join(
                spark.read.format("parquet").load(path_time), 
                "SK_time", 
                "inner"
            )
        max_atribute_gold = df_gold.select(F.max(atribute)).collect()[0][0]
    except:
        df_gold = spark.createDataFrame([], schema)
        max_atribute_gold = None

    return df_gold, max_atribute_gold

def get_max_date(df_silver):
    path_time = f"{silver_path}/dim/time/"
    df_silver = spark.read.format("parquet").load(path_time)\
        .select("SK_time", "created_at")\
        .join(df_silver, "SK_time", "inner")

    min_date_silver = df_silver.select(min('created_at')).collect()[0][0]
    max_date_gold = min_date_silver - timedelta(days=1)
    max_date_gold = max_date_gold.replace(hour=23, minute=59, second=59)

    df_silver.unpersist()
    return max_date_gold

# Atualização tabela tempo
schema_time = StructType([
    StructField("MES", IntegerType(), True),
    StructField("DIA", IntegerType(), True),
    StructField("HORA", IntegerType(), True),
    StructField("MINUTO", IntegerType(), True),
    StructField("DIA_SEMANA", StringType(), True),
    StructField("SK_time", IntegerType(), True),
    StructField("created_at", TimestampType(), True)
])

df_time_gold, max_date_time = get_data_gold(schema_time, time_path_gold, 'SK_time')

if max_date_time is None:
    max_date_time = get_max_date()

df_time_gold = spark.read.option("header", "true") \
        .schema(schema_time)\
        .format("parquet")\
        .load(time_path_silver)\
        .filter(col("created_at") > lit(max_date_gold))

# Atualização tabela author
schema_author = StructType([
    StructField("author", StringType(), True),
    StructField("SK_author", IntegerType(), True)
])

df_author_silver = spark.read.option("header", "true") \
        .schema(schema_author) \
        .format("parquet") \
        .load(author_path_silver)

# Atualização e vetorização tabela posts
schema_gold = StructType([
    StructField("SK_post", LongType(), True),
    StructField("SK_author", LongType(), True),
    StructField("SK_time", LongType(), True),
    StructField("text_cleaned", StringType(), True),
    StructField("like_count", IntegerType(), True),
    StructField("repost_count", IntegerType(), True),
    StructField("quote_count", IntegerType(), True),
    StructField("reply_count", IntegerType(), True)
])

df_silver = spark.read.option("header", "true") \
        .schema(schema_gold) \
        .format("parquet") \
        .load(posts_silver)

df_gold, max_date_gold = get_data_gold(schema_gold, posts_gold, 'created_at')

if max_date_gold is None:
    max_date_gold = get_max_date(df_silver)

df_time_silver = spark.read.format("parquet")\
    .load(time_path_silver)\
    .select("SK_time", "created_at")

df_gold = df_silver.join(df_time_silver, "SK_time", "inner")\
    .filter(col("created_at") > lit(max_date_gold))\
    .drop("created_at")

udf_vector = udf(encode, ArrayType(FloatType()))

df_gold = df_gold\
    .withColumn("embedding", udf_vector(col("yaked"))).drop(col("yaked"))

# Upload camada gold bucket S3
df_time_gold.write.mode("append") \
    .parquet(f"{gold_path}/dim/time/")

df_author_silver.write.mode("overwrite") \
    .option("overwriteSchema", "true")\
    .parquet(f"{gold_path}/dim/author/")

df_gold.write.mode("append") \
    .partitionBy("day_collect") \
    .parquet(f"{gold_path}/fato/posts/")

job.commit()