import importlib.resources
import json
import logging as logger
import multiprocessing as mp
import os
from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd
import yaml
from botocore.exceptions import ClientError
from pymilvus import (
    Collection,
    CollectionSchema,
    FieldSchema,
    utility,
)

from ibm.unifiedsearchvectors.utils.functions import (
    build_cos_parquet_path,
    connect_cos,
    connect_milvus,
    get_schema_comments,
    query_elastic,
)
from ibm.unifiedsearchvectors.utils.html_table_extractor.TableExtractor import (
    TableExtractor,
)

log = logger.getLogger("ibmsearch")

PROCESS_POOL_COUNT = int(os.getenv('PROCESS_POOL_COUNT', 1))


class ParentDoc(ABC):
    def __init__(self, pull_elastic: str):
        self.pull_elastic = pull_elastic

    CHANGE_THRESHOLD: float = 0.0
    DOCS_TYPE: str = ""
    QUERY_FILE_STRING: str = ""
    ADOPTER_SPECIFIC_KEYS = (
        "dwcontenttype",
        "ibmdocstype",
        "ibmdocsproduct",
        "tsdoctypedrill",
        "ibm_tssoftware_version_original",
        "ibmdocskey",
        "tscategory",
    )

    def elastic_cos_pull(self) -> pd.DataFrame:
        with importlib.resources.path(
            "ibm.unifiedsearchvectors.resources", self.QUERY_FILE_STRING
        ) as f_name:
            with f_name.open("rt") as f:
                self.QUERY = json.load(f)

        obj_config = connect_cos()
        base_path = build_cos_parquet_path(self.DOCS_TYPE)
        cos_file_checker = True
        partition_paths = []

        if (self.pull_elastic == "only_pull") or (self.pull_elastic == "both"):
            try:
                prev_len = obj_config.get_parquet_rows(base_path)
                if prev_len == 0:
                    log.info(f"Previous {self.DOCS_TYPE} file length 0, skipping threshold check")
                    cos_file_checker = False
                    new_path = base_path.replace(".parquet", "/")
                    log.info(f"getting partitions from {self.DOCS_TYPE} data from COS at {new_path}")
                    partition_paths = obj_config._list_contents(base_path.replace(".parquet", "/"))
                    log.info(f" partitions to be deleted before the run: {partition_paths}")
                    obj_config.delete_files(partition_paths)
            except ClientError as ex:
                if ex.response['Error']['Code'] == 'NoSuchKey':
                    log.info(f"No {self.DOCS_TYPE} file in COS, skipping threshold check")
                    cos_file_checker = False
                else:
                    raise ex

            def partition_writer(part_num, table):
                part_path = base_path.replace(".parquet", f"/part-{part_num:04d}.parquet")
                obj_config.insert_parquet_pyarrow(tb=table, file_path=part_path)
                partition_paths.append(part_path)
                log.info(f"Wrote partition {part_num} to {part_path}")

            query_elastic(
                self.QUERY,
                self.clean_pull,
                adopter_specific_keys=self.ADOPTER_SPECIFIC_KEYS,
                partition_callback=partition_writer,
                # partition_batch_size=5
            )

            curr_len = sum(obj_config.get_parquet_rows(path) for path in partition_paths)
            if cos_file_checker and (
                abs(curr_len - prev_len) / prev_len > float(os.getenv('CHANGE_THRESHOLD', self.CHANGE_THRESHOLD))
            ):
                raise Exception(f"{self.DOCS_TYPE} surpassed change threshold!!!")

            log.info("Elastic pull written to COS in partitions")
        else:
            log.info(f"Reading existing {self.DOCS_TYPE} data from COS at {base_path}")
            # List all partitioned files if base_path is a prefix
            # partition_paths = obj_config.list_partition_paths(base_path.replace(".parquet", "/"))
        new_path = base_path.replace(".parquet", "/")
        log.info(f"getting partitions from {self.DOCS_TYPE} data from COS at {new_path}")
        partition_paths = obj_config._list_contents(base_path.replace(".parquet", "/"))
        log.info(f" partitions are {partition_paths}")
            

        # Read all partitions and concatenate
        all_dfs = [obj_config.read_parquet(file_name=path) for path in partition_paths]
        return pd.concat(all_dfs, ignore_index=True)



    @abstractmethod
    def clean_pull(self, sk_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_schema_fields(self, fields_dict: dict, dim: int) -> list[FieldSchema]:
        raise NotImplementedError

    def internal_entitled_handler(self, sk_df: pd.DataFrame):
        # the default behavior would be to take non-internal and non entitled
        # subclasses can override
        sk_df['entitled'] = sk_df['entitled'].astype(str)
        if "internal_only" in sk_df.columns:
            sk_df['internal_only'] = (
                sk_df['internal_only'].fillna(0).astype(bool)
            )  # astype turns nans to true by default so fill with 0, not False or we'll get cast warnings.
            sk_df = sk_df[sk_df['internal_only'] != True]
        sk_df = sk_df[sk_df["entitled"] != "1"]
        sk_df['entitled'] = False
        return sk_df

    def content_is_title_filter(self, sk_df: pd.DataFrame) -> pd.DataFrame:
        """
        This function filters out rows from a pandas DataFrame where the title and body content are the same.
        It takes a pandas DataFrame as input and returns a filtered DataFrame.

        Args:
            sk_df (pd.DataFrame): The input DataFrame to be filtered.

        Returns:
            pd.DataFrame: The filtered DataFrame with rows where title and body content are the same removed.
        """
        if "body" in sk_df.columns:
            sk_df = sk_df[
                sk_df["title"].str.lower().str.strip()
                != sk_df["body"].str.lower().str.strip()
            ]
        elif "raw_body" in sk_df.columns:
            sk_df = sk_df[
                sk_df["title"].str.lower().str.strip()
                != sk_df["raw_body"].str.lower().str.strip()
            ]
        return sk_df

    def perform_table_extract(self, data: pd.DataFrame):
        with importlib.resources.path(
            "ibm.unifiedsearchvectors.resources", "tables_config.yaml"
        ) as f_name:
            with f_name.open("rt") as f:
                config = yaml.safe_load(f)

        html_table_extractor = TableExtractor(config)

        if PROCESS_POOL_COUNT > 1:
            log.info(f"table extraction with {PROCESS_POOL_COUNT} processes")
            with mp.Pool(PROCESS_POOL_COUNT) as pool:
                data['tables'] = pool.map(
                    html_table_extractor.extract_html_tables, data["content"]
                )
        else:
            log.info("table extraction starting")
            data["tables"] = data["content"].apply(
                html_table_extractor.extract_html_tables
            )
        log.info("table extraction complete")

    def create_collection(self, collection_name: str, dim: int) -> Collection:
        connect_milvus()
        if utility.has_collection(collection_name):
            return Collection(name=collection_name)

        # grab the comments from the config to merge into the schema
        comments_dict: dict = get_schema_comments(self.DOCS_TYPE)
        fields_dict = comments_dict.get('fields', {})

        fields = self.get_schema_fields(fields_dict, dim=dim)
        if os.getenv('MODEL_NAME').startswith('ibm/slate-30m-english-rtrvr'):
            display_model_name = 'ibm/slate-30m-english-rtrvr-v2'
        else:
            display_model_name = os.getenv('MODEL_NAME')
        schema = CollectionSchema(
            fields=fields,
            description="DESCRIPTION: "
            + comments_dict.get("collection", '')
            + "\n"
            + f"EMBEDDING_MODEL: {display_model_name}\n"
            + "CHUNKING_SIZE: 2000\n"
            + "CHUNKING_OVERLAP: 200\n"
            + "LAST_UPDATE_DATE: "
            + datetime.now().strftime("%Y_%m_%d"),
        )
        prop = {"collection.ttl.seconds": 0}
        collection = Collection(
            name=collection_name, schema=schema, properties=prop, num_shards=2
        )

        index_params = {
            'metric_type': "IP",
            'index_type': "IVF_FLAT",
            'params': {"nlist": 2048},
        }
        collection.create_index(field_name='doc_vector', index_params=index_params)
        log.info(
            "Created a collection '{}' with {} dimensions.".format(collection_name, dim)
        )
        return collection