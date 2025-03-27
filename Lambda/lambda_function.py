import boto3
import csv
from io import StringIO
from atproto import Client
from datetime import datetime



CREDENTIAL = 'user.credential'
PASSWORD = 'user.password'
BUCKET = 'bsky-posts-lake'

date_time_now = datetime.now()
date_time = date_time_now.strftime("%Y-%m-%d_%H-%M-%S")

client = Client()

def get_posts():
  feed_response = client.app.bsky.feed.get_feed(
      {'feed': 'at://did:plc:z72i7hdynmk6r22z27h6tvur/app.bsky.feed.generator/whats-hot', 
        'limit': 100,
        'filter': ['posts_no_replies', 'posts_no_media']},
      headers={'Accept-Language': 'en'})

  return feed_response

def create_csv(feed):
  feed_dict = {'author': [item.post.author.handle for item in feed.feed],
             'text': [item.post.record.text for item in feed.feed],
             'created_at': [item.post.record.created_at for item in feed.feed],
             'like_count': [item.post.like_count for item in feed.feed],
             'repost_count': [item.post.repost_count for item in feed.feed],
             'quote_count': [item.post.quote_count for item in feed.feed],
             'reply_count': [item.post.reply_count for item in feed.feed]}

  feed_buffer = StringIO()
  csv_writer = csv.writer(feed_buffer)

  csv_writer.writerow(list(feed_dict.keys()))
  csv_writer.writerows(zip(*feed_dict.values()))

  return feed_buffer

def lambda_handler(event, context):
  client.login(CREDENTIAL, PASSWORD)

  feed_en = get_posts()

  buffer_en = create_csv(feed_en)

  s3 = boto3.client('s3')
  try:
    s3.put_object(Bucket=BUCKET, Key=f'raw/en/{date_time}_en.csv', Body=buffer_en.getvalue())
  except Exception as e:
    print(e)

  return {"statusCode": 200, "body": f"Arquivos salvos."}