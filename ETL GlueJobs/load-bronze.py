import boto3
import re
from pyspark.sql.types import *
from pyspark.sql import functions as F

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
bucket_name = "bsky-posts-lake"
prefix = "raw/en/unprocessed/"

s3_client = boto3.client("s3")
response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
object_list = response.get("Contents", [])

files_by_date = {}
date_pattern = re.compile(r"(\d{4}-\d{2}-\d{2})")

for obj in response.get("Contents", []):
    key = obj["Key"]
    match = date_pattern.search(key)
    
    if match:
        date_str = match.group(1)
        if date_str not in files_by_date:
            files_by_date[date_str] = []
        files_by_date[date_str].append(key)

if files_by_date:
    oldest_files = files_by_date[min(files_by_date.keys())]
else:
    oldest_files = None

schema_raw = StructType([
    StructField("author", StringType(), True),
    StructField("text", StringType(), True),
    StructField("created_at", TimestampType(), True),
    StructField("like_count", IntegerType(), True),
    StructField("repost_count", IntegerType(), True),
    StructField("quote_count", IntegerType(), True),
    StructField("reply_count", IntegerType(), True)
])

if oldest_files:
    try:
        s3_input_paths = [f"s3://{bucket_name}/{file}" for file in oldest_files]
        df = spark.read.option("multiLine", "true") \
            .option("delimiter", ",") \
            .option("quote", "\"") \
            .option("escape", "\"") \
            .option("header", "true") \
            .schema(schema_raw) \
            .csv(s3_input_paths)
    except Exception as e:
        print(f"Fail to load {raw_path_in}: {e}")
        
    s3_bronze_path = "s3://bsky-posts-lake/bronze/"

    df.withColumn("day_collect", F.dayofmonth("created_at")) \
        .write.mode("append") \
        .partitionBy("day_collect") \
        .parquet(s3_bronze_path)
    
    for file_path in s3_input_paths:
    new_path = file_path.replace("unprocessed", "processed")

    s3_client.copy_object(
        Bucket=bucket_name,
        CopySource={"Bucket": bucket_name, "Key": file_path},
        Key=new_path)
    s3_client.delete_object(Bucket=bucket_name, Key=file_path)

else:
    pass

job.commit()