# %%
import nltk
import re
import json
import unicodedata
import string
import boto3
from pyspark.sql import functions as F
from nltk.corpus import stopwords
from pyspark.sql.types import StringType

# %%
ACCESS_KEY = 'S3_ACCESS_KEY'
SECRET_KEY = 'S3_SECRET_KEY'

spark.conf.set("fs.s3a.access.key", ACCESS_KEY)
spark.conf.set("fs.s3a.secret.key", SECRET_KEY)
spark.conf.set("fs.s3a.endpoint", "s3.us-east-2.amazonaws.com")
spark.conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3.S3AFileSystem")
spark.conf.set("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")

# %%
nltk.download('stopwords')

def collect_language(lang):
    s3 = boto3.client(
        's3',
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY
    )
    response = s3.list_objects_v2(Bucket='S3-name', Prefix=f'raw/{lang}/')
    csv_files = [
        f"https://S3-name.s3.us-east-2.amazonaws.com/{content['Key']}"
        for content in response.get('Contents', [])
    ]
    return csv_files[1:]

# %%
destination_path = "dbfs:/mnt/S3-name/raw/pt/"
dbutils.fs.mkdirs(destination_path)

for doc in collect_language('en'):
    print(doc)
    dbutils.fs.cp(doc, f"dbfs:/mnt/S3-name/raw/pt")

# %%
with open("add_stopwords.json", "r", encoding="utf-8") as file:
    more_stops = json.load(file)

# %%
# Cleaning Functions

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
    text = re.compile(r'[-+]?[.\d]*\d+[:,./\d]*').sub(r'', text) # sinais e pontos em numeros
    text =  re.compile(r'https?://\S+|www\.\S+').sub(r'URL',text) #links
    text = re.compile(r'([!?.]){1,}').sub(r'\1 ', text) # sinais repetidos
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

# %%
def normalizing_text(text):
    def is_normal_character(char):
        return char in string.ascii_letters or char in string.digits

    text_normal = all(is_normal_character(char) for char in unicodedata.normalize('NFKC', text))
    return text_normal

def remove_stopwords(text):
    stops = list(stopwords.words('english'))
    not_apostrophe = [word.replace("'", "") for word in stops if "'" in word]
    stops = set(stops + not_apostrophe).union(more_stops['pronoum_stopwords'],
                                              more_stops['extra_stopwords'],
                                              more_stops['internet_stopwords'])
    text = ' '.join([word for word in text.split() if word not in stops and len(word) >= 3])
    return text

# %%
# concatenar todos os csvs em um dataset só
df = spark.read.option("multiLine", "true") \
    .option("delimiter", ",") \
    .option("quote", "\"") \
    .option("escape", "\"") \
    .option("header", "true") \
    .csv("dbfs:/mnt/S3-name/raw/pt") \
    .drop_duplicates(["created_at", "text", "author"]) \
    .dropna(subset=["text"])

# %%
udf_clean = udf(cleaning_posts, StringType())
udf_stop = udf(remove_stopwords, StringType())
udf_normal = udf(normalizing_text, StringType())

df = df.withColumn("cleaned", udf_clean(df["text"])).drop(df.text)
df = df.withColumn("stopped", udf_stop(df.cleaned)).drop(df.cleaned)
df = df.withColumn("normal", udf_stop(df.stopped)).drop(df.stopped)
df = df.filter(F.col("normal").isNotNull() & (F.trim(F.col("normal")) != ""))

# %%
df.coalesce(1).write.csv("dbfs:/FileStore/tables/bsky_posts.csv", header=True, mode="overwrite")

# %%
path = "dbfs:/FileStore/tables/bsky_posts.csv/part-00000-tid-5322256841719065438-96f70029-7917-49a2-8a4d-f480e8c3d045-606-1-c000.csv"
arquivo = "part-00000-tid-5322256841719065438-96f70029-7917-49a2-8a4d-f480e8c3d045-606-1-c000.csv"

dbutils.fs.cp(path, f"file:/tmp/{arquivo}")

# %%
s3_client = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
s3_client.upload_file(f"/tmp/{arquivo}", "S3-name", f"raw/{arquivo}")
