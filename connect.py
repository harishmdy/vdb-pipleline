"""Utilities for COS bucket connections and transformations."""

import json
import logging
import os
import pickle
import re
from datetime import datetime
from io import BytesIO, StringIO
from typing import Optional, cast

import boto3
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import s3fs
from botocore.config import Config
from botocore.exceptions import ClientError
from mypy_boto3_s3 import S3Client
from pyarrow import fs
from tqdm import tqdm

from ibm.unifiedsearchvectors.utils.carrot.api_connector import ApiConnectorBase
from ibm.unifiedsearchvectors.utils.carrot.configuration import SimpleConfigLoader
from ibm.unifiedsearchvectors.utils.config import (
    ObjectStorageConfig,
    ObjectStorageDictConfig,
)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
class CosManager(ApiConnectorBase):
    """
    Create an instance of COS Manager.
    Please use the following constructor:
    - `CosManager.create_from_dict`
    - `CosManager.create_from_config`
    """


    def __init__(self, config: ObjectStorageConfig) -> None:
        self.logger = logging.getLogger(__name__)

        self.access_key_id = config.access_key
        self.secret_access_key = config.secret_key
        self.bucket = config.bucket
        self.endpoint_url = config.endpoint_url

        self.conn = self.setup()

    def is_setup(self):
        return self.conn is not None

    def setup(self) -> S3Client:
        config: Config = Config(
            connect_timeout=360, retries={"max_attempts": 10, "mode": "standard"}
        )

        self.__connection = cast(
            S3Client,
            boto3.client(
                "s3",
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                endpoint_url=self.endpoint_url,
                config=config,
            ),
        )

        return self.__connection

    # def _list_contents(
    #     self,
    #     max_keys: int = 1000,
    # ) -> list[str]:
    #     """Get list of bucket file names."""

    #     file_list = []
    #     next_token: Optional[str] = None

    #     logging.info('Getting list of bucket contents...')

    #     while True:
    #         if next_token:
    #             obj_list = self.conn.list_objects_v2(
    #                 Bucket=self.bucket, MaxKeys=max_keys, ContinuationToken=next_token
    #             )
    #         else:
    #             obj_list = self.conn.list_objects_v2(
    #                 Bucket=self.bucket, MaxKeys=max_keys
    #             )

    #         batch_files = [obj["Key"] for obj in obj_list["Contents"] if 'Key' in obj]
    #         file_list.extend(batch_files)

    #         if obj_list["IsTruncated"]:
    #             next_token = obj_list["NextContinuationToken"]
    #         else:
    #             return file_list




    def insert_file_stream(self, buf: BytesIO, file_path: str, content_type: str = "application/octet-stream"):
        """
        Uploads a BytesIO stream to the specified path in COS.

        :param buf: BytesIO stream containing the file data.
        :param file_path: Target path in the COS bucket.
        :param content_type: MIME type for the file.
        """
        if not hasattr(buf, "seek") or not hasattr(buf, "read"):
            raise ValueError("Expected a file-like object with seek and read methods")

        buf.seek(0)  # Reset buffer position
        try:
            self.conn.upload_fileobj(
                Fileobj=buf,
                Bucket=self.bucket,
                Key=file_path,
                ExtraArgs={"ContentType": content_type}
            )
            self.logger.info(f"Uploaded stream to COS at {file_path}")
        except ClientError as e:
            self.logger.error(f"Failed to upload {file_path} to COS: {e}")
            raise
    

    def download_fileobj(self, file_name: str, buf: BytesIO):
        """
        Downloads the file from COS and stores it in the provided buffer.
        
        :param file_name: The name of the file to download from COS.
        :param buf: The buffer to store the downloaded file data.
        """
        self.conn.download_fileobj(self.bucket, file_name, buf)


    def _list_contents(
        self,
        prefix: str = "",
        max_keys: int = 1000,
    ) -> list[str]:
        """Get list of bucket file names."""

        file_list = []
        next_token: Optional[str] = None

        logging.info('Getting list of bucket contents...')

        while True:
            if next_token:
                obj_list = self.conn.list_objects_v2(
                    Bucket=self.bucket,MaxKeys=max_keys, ContinuationToken=next_token,Prefix=prefix
                )
            else:
                obj_list = self.conn.list_objects_v2(
                    Bucket=self.bucket, MaxKeys=max_keys,Prefix=prefix
                )

            # Safe access: if no files, 'Contents' key may be absent
            if "Contents" in obj_list:
                batch_files = [obj["Key"] for obj in obj_list["Contents"] if "Key" in obj]
                file_list.extend(batch_files)
            else:
                break  # Nothing to list

            # batch_files = [obj["Key"] for obj in obj_list["Contents"] if 'Key' in obj]
            # file_list.extend(batch_files)

            

            if obj_list.get("IsTruncated"):
                next_token = obj_list.get("NextContinuationToken")
            else:
                break
        return file_list

    def download(self, file: str):
        """
        Helper function to pull an object from COS (also required DbManagerBase
        function).
        """
        return self.conn.get_object(Bucket=self.bucket, Key=file)["Body"].read()

    def _read_csv(self, file_name: str) -> pd.DataFrame:
        """Download csv file from S3 & load into Pandas dataframe."""

        bytes = self.download(file_name)
        data_string = str(bytes, "utf-8")
        data = StringIO(data_string)

        df = pd.read_csv(data, dtype=object)

        return df

    def read_feather(self, file_name: str) -> pd.DataFrame:
        """Download feather file from S3 & load into Pandas dataframe."""
        return pd.read_feather(BytesIO(self.download(file_name)))

    def read_parquet(self, file_name: str) -> pd.DataFrame:
        """Download parquet file from S3 & load into Pandas dataframe."""
        return pd.read_parquet(BytesIO(self.download(file_name)))

    def read_parquet_multiparts(self, parquet_keys: list[str]) -> pd.DataFrame:
        dfs = [self.read_parquet(file_name=key) for key in parquet_keys]
        return pd.concat(dfs, ignore_index=True)
    
    def read_pickle(self, file_name: str):
        """Download pickled file from COS bucket."""

        obj = self.download(file_name)

        bytes = BytesIO(obj)
        data = pickle.load(bytes)

        return data

    def read_json(self, file_name: str) -> pd.DataFrame:
        """Download json file from S3 & load into Pandas dataframe."""
        return pd.read_json(BytesIO(self.download(file_name)))

    def aggregate_files(self, file_names: list[str]) -> pd.DataFrame:
        """Aggregate COS bucket files by regex matching file names."""

        all_files = self._list_contents()
        matched_files = []

        logging.info('Creating file regex patterns...')

        for file in all_files:
            for pattern in file_names:
                if re.match(pattern, file):
                    matched_files.append(file)

        print("Pulling and concatenating dataframes...")
        data = [self._read_csv(file) for file in tqdm(matched_files)]

        return pd.concat(data, ignore_index=True, axis=0)

    def insert(self, df: pd.DataFrame, file_path: str, file_type: str) -> None:
        """Upload a file to COS bucket."""
        buffer = BytesIO()
        if file_type == "json":
            df.to_json(buffer, orient="records")
        else:
            df.to_parquet(buffer)
        self.conn.put_object(Bucket=self.bucket, Body=buffer.getvalue(), Key=file_path)

    def insert_parquet_pyarrow(self, tb: pyarrow.Table, file_path: str) -> None:
        s3 = s3fs.S3FileSystem(
            anon=False,
            key=self.access_key_id,
            secret=self.secret_access_key,
            client_kwargs={'endpoint_url': self.endpoint_url},
        )
        pq.write_table(tb, f's3://{self.bucket}/{file_path}', filesystem=s3)

    def delete_files(self, file_names: list[str]) -> None:
        """Delete a file, or a list of files from COS bucket"""

        for file in file_names:
            self.conn.delete_object(Bucket=self.bucket, Key=file)
        return

    def get_parquet_rows(self, file_path: str) -> int:
        s3 = s3fs.S3FileSystem(
            anon=False,
            key=self.access_key_id,
            secret=self.secret_access_key,
            client_kwargs={'endpoint_url': self.endpoint_url},
        )
        try:
            meta = pq.read_metadata(
                f's3://{self.bucket}/{file_path}',
                filesystem=s3,
            )
            return meta.num_rows
        except FileNotFoundError:
            logging.info(f"{file_path} not found on COS.")
            return 0
    

    @classmethod
    def create_from_config(cls, file_path: str, bucket: str) -> 'CosManager':
        """
        Create COS Manager from a config file.
        Sample file
            ```yaml
            # cos.yaml
            credentials: # find credentials under service credentials
                access_key: '1234asdfjwekd...'
                secret_key: '1234asdfjwekd....::'
            buckets:
                logs: # bucket alias
                    bucket_name: 'my-very-unique-storage-bucket'
                    endpoint_url: 'https://s3.us-south.cloud-object-storage.appdomain.cloud'
                data:
                    bucket_name: 'all-my-training-data-from-project-abc-bucket'
                    endpoint_url: 'https://s3.us-east.cloud-object-storage.appdomain.cloud'
            ```
        Args:
            file_path (str): path to the config file
            bucket (str): the bucket alias in the config file
        Returns:
            CosManager instance
        Raises:
            Exception: if config doesn't exist
        """

        cfg = SimpleConfigLoader(file_path).get()

        if not cfg:
            raise Exception('No config found!')

        creds_cfg: dict[str, str] = cfg.get('credentials', dict())
        buckets_cfg: dict[str, dict[str, str]] = cfg.get('buckets', dict())
        bucket_cfg = buckets_cfg.get(bucket, dict())

        config = ObjectStorageConfig.validate({**creds_cfg, **bucket_cfg})

        return cls(config)

    @classmethod
    def create_from_dict(cls, config: ObjectStorageDictConfig) -> 'CosManager':
        """
         Create COS Manager from a dictionary (**NOT Recommended**)
        !!! danger "Attention"
            Please make sure that you do **NOT** commit the api_key to GitHub.
        !!! example "Sample dictionary"
        ```
        {
            'access_key': '1234asdfjwekd...',
            'secret_key': '1234asdfjwekd....::',
            'bucket_name': 'my-very-unique-storage-bucket'
            'endpoint_url': 'https://s3.us-south.cloud-object-storage.appdomain.cloud'
        }
        ```
        """

        cos_cfg = ObjectStorageConfig.validate(config)

        return cls(cos_cfg)

    def query_logs(
        self, earliest_date: datetime, latest_date: datetime
    ) -> pd.DataFrame:
        unified_log_names = get_unified_log_names(earliest_date, latest_date)
        new_logs = self.aggregate_files(file_names=unified_log_names)

        return new_logs

    # Added to download folders from COS
    def download_folder_from_cos(self, folder_prefix: str, local_dir: str) -> None:
        """
        Downloads an entire folder from COS to a local directory, including directories.

        Args:
            folder_prefix (str): The prefix of the folder to be downloaded from COS.
            local_dir (str): Path to the local directory where the folder will be downloaded.
        """
        # Ensure the local directory exists
        os.makedirs(local_dir, exist_ok=True)

        # Initialize paginator
        paginator = self.conn.get_paginator('list_objects_v2')

        # Create a list to store all object keys
        all_keys = []

        # Iterate over pages
        for page in paginator.paginate(Bucket=self.bucket, Prefix=folder_prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    all_keys.append(obj['Key'])

        if not all_keys:
            logging.info(f"No objects found with prefix '{folder_prefix}'")
            return

        # Download each object
        for obj_key in all_keys:
            # Determine the local file or directory path
            relative_path = os.path.relpath(obj_key, folder_prefix).replace("/", os.sep)
            local_path = os.path.join(local_dir, relative_path)

            # Check if the key represents a directory
            if obj_key.endswith('/'):
                # Create the local directory
                os.makedirs(local_path, exist_ok=True)
                logging.info(f'Created directory {local_path}')
            else:
                # Create the local directory if it doesn't exist
                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                logging.info(f'Downloading {obj_key} to {local_path}...')

                with open(local_path, 'wb') as f:
                    self.conn.download_fileobj(self.bucket, obj_key, f)

    def insert_json(self, json_object, file_path):
        self.conn.put_object(Bucket=self.bucket, Body=json.dumps(json_object, cls=NpEncoder), Key=file_path)
