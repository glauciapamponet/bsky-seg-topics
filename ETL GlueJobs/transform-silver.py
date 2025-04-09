import nltk
import re
import unicodedata
import shutil
import string
import enchant
import yake
import boto3
import numpy as np
import wordninja
import pyspark.pandas as ps
from pyspark.sql import functions as F
from pyspark.ml.feature import Tokenizer
from functools import reduce
from nltk.corpus import stopwords
from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, lit

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
s3 = boto3.client("s3")

bucket_name = "bsky-posts-lake"
bronze_path = "s3://bsky-posts-lake/bronze"
silver_path = "s3://bsky-posts-lake/silver"
add_json = "resources/add_stopwords.json"
nltk_json = "resources/nltk_stopwords.json"

def get_stopwords(path):
    obj = s3.get_object(Bucket=bucket_name, Key=path)
    json_content = obj["Body"].read().decode("utf-8")
    return json.loads(json_content)

def cleaning_posts(text):
    signs = "!@#$%^&*()_+-=,.<>?/;:'\"[]{}|`~…“”’‘"
    emoji_pattern = re.compile("[" +
        u"\U0001F600-\U0001F64F"  # Emoticons
        u"\U0001F300-\U0001F5FF"  # Símbolos e pictogramas
        u"\U0001F680-\U0001F6FF"  # Transporte e mapas
        u"\U0001F700-\U0001F77F"  # Alquimia
        u"\U0001F780-\U0001F7FF"  # Geométricos adicionais
        u"\U0001F800-\U0001F8FF"  # Suplementos
        u"\U0001F900-\U0001F9FF"  # Suplementos adicionais
        u"\U0001FA00-\U0001FA6F"  # Objetos diversificados
        u"\U0001FA70-\U0001FAFF"  # Objetos adicionais
        u"\U00002600-\U000027BF"  # Símbolos miscelâneos
        u"\U0001F1E6-\U0001F1FF"  # Bandeiras regionais
        u"\U0001F201-\U0001F251"  # Símbolos diversos
        u"\U00002B06-\U00002B07"
        u"\U0001F200-\U0001F251"
        u"\U0000203C-\U000032FF"
        u"\U00002702-\U000027B0"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.compile(r'@\S+').sub(r'',text) #mentions
    text = re.compile(r'<.*?>').sub(r'',text) #html tags
    text = re.compile(r'[-+]?[.\d]*\d+[:,./\d]*').sub(r'', text) #number signs and puncts
    text =  re.compile(r'https?://\S+|www\.\S+').sub(r'URL',text) #links
    text = re.compile(r'([!?.]){1,}').sub(r'\1 ', text) #sign repetitions
    smiley = re.compile(r'[8:=;][\'\-]?[)dDp]') #emojis
    text = smiley.sub(r'', text)
    smiley = re.compile(r'[8:=;][\'\-]?[(\\/]') #emojis
    text = smiley.sub(r'', text)
    text = re.compile(r'<3').sub(r'HEART', text) #emojis
    text = re.compile(r'[?!:,./\"“”’‘();_#+&*$@…»|]').sub(r'', text) 
    text = re.compile(r"[•'—\[\]%]").sub(r'', text)
    text = re.sub(r"’s\b", "", text)
    text = re.sub(r"'s\b", "", text)
    text = re.compile(r'\b(\S*?)([a-z])\2{2,}\b').sub(r'\1\2 ', text)
    text = re.sub(' - ', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = ' '.join([word.strip(signs) for word in text.split(" ")])
    text = text.lower()
    return text

def normalizing_text(text):
    def is_normal_character(char):
        return char in string.ascii_letters or char in string.digits

    text_normal = all(is_normal_character(char) for char in unicodedata.normalize('NFKC', text))
    return text_normal

def remove_stopwords(text):
    more_stops = get_stopwords(add_json)
    stops = list(get_stopwords(nltk_json))
    not_apostrophe = [word.replace("'", "") for word in stops if "'" in word]
    stops = set(stops + not_apostrophe).union(more_stops['pronoum_stopwords'],
                                              more_stops['extra_stopwords'],
                                              more_stops['internet_stopwords'])
    text = ' '.join([word for word in text.split() if word not in stops and len(word) >= 3])
    return text

def separate_words(text):
    separated_text = list()
    for word in text.split(" "):
        separated_text += wordninja.split(word)
    text = " ".join(separated_text)
    return text

extractor = yake.KeywordExtractor(lan="en",          
                                  n=3,               
                                  dedupLim=0.85,      
                                  dedupFunc='seqm',  
                                  windowsSize=2,     
                                  top=15)
def yakinizer(text):
    seen = set()
    key_words = extractor.extract_keywords(text)
    filtered_words = ' '.join([kw for kw, sc in key_words if sc < 0.05])
    filtered_words = filtered_words.split(' ')
    final_words = [x for x in filtered_words if not (x in seen or seen.add(x))]
    return ' '.join(final_words)

def get_data_silver(schema, path, atribute):
    try: 
        df_silver = spark.read.format("parquet") \
            .option("header", "true") \
            .schema(schema) \
            .load(path)
        max_atribute_silver = df_silver.select(F.max(atribute)).collect()[0][0]
    except:
        df_silver = spark.createDataFrame([], schema)
        max_atribute_silver = None

    return df_silver, max_atribute_silver

schema_bronze = StructType([
    StructField("author", StringType(), True),
    StructField("text", StringType(), True),
    StructField("created_at", TimestampType(), True),
    StructField("like_count", IntegerType(), True),
    StructField("repost_count", IntegerType(), True),
    StructField("quote_count", IntegerType(), True),
    StructField("reply_count", IntegerType(), True)
])

df_bronze = spark.read.format("parquet") \
    .option("header", "true") \
    .schema(schema_bronze) \
    .load(bronze_path)

schema_silver = StructType([
    StructField("SK_post", IntegerType(), True),
    StructField("SK_author", IntegerType(), True),
    StructField("text_cleaned", StringType(), True),
    StructField("SK_time", IntegerType(), True),
    StructField("like_count", IntegerType(), True),
    StructField("repost_count", IntegerType(), True),
    StructField("quote_count", IntegerType(), True),
    StructField("reply_count", IntegerType(), True)
])

posts_silver_path = f"{silver_path}/fato"
df_silver, max_date_silver = get_data_silver(schema_silver, posts_silver_path, 'created_at')

if max_date_silver is None:
    min_date_bronze = df_bronze.select(F.min('created_at')).collect()[0][0]
    max_date_silver = min_date_bronze - timedelta(days=1)
    max_date_silver = max_date_silver.replace(hour=23, minute=59, second=59)

df_silver = df_bronze.filter(F.col("created_at") > max_date_silver)\
    .drop_duplicates(["created_at", "text", "author"])\
    .dropna(subset=["text"])\
    .withColumn("created_at", F.date_format("created_at", "yyyy-MM-dd HH:mm"))

udf_clean = udf(cleaning_posts, StringType())
udf_stop = udf(remove_stopwords, StringType())
udf_normal = udf(normalizing_text, StringType())
udf_separate = udf(separate_words, StringType())
udf_yake = udf(yakinizer, StringType())

df_silver = df_silver.withColumn("cleaned", udf_clean(F.col("text"))).drop(F.col("text"))\
    .withColumn("stopped", udf_stop(F.col("cleaned"))).drop(F.col("cleaned"))\
    .withColumn("normal", udf_stop(F.col("stopped"))).drop(F.col("stopped"))\
    .withColumn("text_cleaned", udf_separate(F.col("normal"))).drop(F.col("normal"))\
    .withColumn("yaked", udf_yake(F.col("text_cleaned"))).drop(F.col("text_cleaned"))\
    .filter(F.trim(F.col("yaked")) != "")\
    .dropna(subset=["yaked"])\
    
schema_time = StructType([
    StructField("MES", IntegerType(), True),
    StructField("DIA", IntegerType(), True),
    StructField("HORA", IntegerType(), True),
    StructField("MINUTO", IntegerType(), True),
    StructField("DIA_SEMANA", IntegerType(), True),
    StructField("SK_time", IntegerType(), True),
    StructField("created_at", TimestampType(), True)
])

tempo_path = f"{silver_path}/dim/tempo"
df_time_silver, max_SK_time = get_data_silver(schema_time, tempo_path, 'SK_time')

if max_SK_time is None:
    max_SK_time = 0

df_time_silver = df_silver.alias("bronze").select("created_at")\
    .join(df_time_silver.alias("silver"), 'created_at', "left_anti")\
    .select("created_at").orderBy("created_at")\
    .drop_duplicates()\
    .withColumn("MES", F.month(F.col("created_at")))\
    .withColumn("DIA", F.dayofmonth(F.col("created_at")))\
    .withColumn("HORA", F.hour(F.col("created_at")))\
    .withColumn("MINUTO", F.minute(F.col("created_at")))\
    .withColumn("DIA_SEMANA", F.date_format(F.col("created_at"), "EEEE"))\
    .withColumn("SK_time", F.monotonically_increasing_id()+max_SK_time+1)

df_author_silver.write.mode("append") \
        .parquet(tempo_path)

author_path = f"{silver_path}/dim/author"
schema_author = StructType([
    StructField("author", StringType(), True),
    StructField("SK_author", IntegerType(), True)
])

df_author_silver, max_SK_author = get_data_silver(schema_author, author_path, 'SK_author')

if max_SK_author is None:
    max_SK_author = 0

df_author_silver = df_silver.alias("bronze").select("author")\
    .join(df_author_silver.alias("silver"), 'author', "left_anti")\
    .select("author")\
    .drop_duplicates(["author"])\
    .withColumn("SK_author", F.monotonically_increasing_id()+max_SK_author+1)

df_author_silver.write.mode("append") \
        .parquet(author_path)

df_silver = df_silver.join(df_time_silver.select("created_at", "SK_time"), on="created_at", how="inner")\
    .join(df_author_silver, on="author", how="inner")\
    .drop("created_at", "author")\
    .withColumn("SK_post", F.monotonically_increasing_id()+1)

df_silver.write.mode("append") \
        .partitionBy("day_collect") \
        .parquet(f"{silver_path}/posts")

job.commit()