import s3fs
import pandas as pd

class LoadingData():
    def __init__(self):
        pass
        self.__BUCKET_PATH = "s3://bsky-posts-lake"

        self.__DIRECTORY = {
            "silver": {
                "posts": f"{self.__BUCKET_PATH}/silver/fato/posts",
                "author": f"{self.__BUCKET_PATH}/silver/dim/author",
                "time": f"{self.__BUCKET_PATH}/silver/dim/time",
            },
            "gold": {
                "posts": f"{self.__BUCKET_PATH}/gold/fato/posts",
                "author": f"{self.__BUCKET_PATH}/gold/dim/author",
                "time": f"{self.__BUCKET_PATH}/gold/dim/time",
            }
        }

    def _loading_table(self, path):
        s3 = s3fs.S3FileSystem()
        parquet_files = s3.glob(path)
        df_posts = pd.DataFrame()
        for file in parquet_files:
            with s3.open(file) as f:
                df_posts = pd.concat([df_posts, pd.read_parquet(f)])
        return df_posts
    
    def load_posts(self, layer="gold"):
        return self._loading_table(f"{self.__DIRECTORY[layer]['posts']}/*/*.parquet")
    
    def load_author(self, layer="gold"):
        return self._loading_table(f"{self.__DIRECTORY[layer]['author']}/*.parquet")
    
    def load_time(self, layer="gold"):
        return self._loading_table(f"{self.__DIRECTORY[layer]['time']}/*.parquet")

