import ast
import logging as logger
import time

# from collections.abc import Iterator
from typing import Generator

import pandas as pd
from pymilvus import (
    DataType,
    FieldSchema,
)

from ibm.unifiedsearchvectors.data_builder.parent_doc_builder import ParentDoc
from ibm.unifiedsearchvectors.utils.functions import (
    cast_to_list,
    convert_datetime,
    truncate_list,
    truncate_list_elements,
)

log = logger.getLogger("ibmsearch")


class IbmDocs(ParentDoc):
    DOCS_TYPE: str = 'ibm_docs'

    # Define keys for each collection
    METADATA_KEYS = [
        "title",
        "body",
        "url",
        "description",
        "entitled",
        "language",
        "dcdate",
        "ibmentitlement",
        "scope",
        "mhscope",
        "dcc",
        "field_keyword_01",
        "latest_version",
        "semver_tags",
        "altver_tags",
        "mtm_tags",
        # "ibmdocstype",
        # "ibmdocsproduct",
    ]

    ADOPTER_SPECIFIC_KEYS = (
        "ibmdocstype",
        "ibmdocsproduct",
        "ibmdocskey",
    )

    CONTENT_KEYS = ["content"]

    CHANGE_THRESHOLD: float = 0.3
    QUERY_FILE_STRING: str = "ibm_docs_query.json"

    QUERY = None

    def clean_pull(self, sk_df: pd.DataFrame) -> pd.DataFrame:
        sk_df = self.internal_entitled_handler(sk_df)
        sk_df = self.content_is_title_filter(sk_df)

        sk_df["adopter_specific"] = sk_df["adopter_specific"].apply(
            lambda x: {k: x.get(k, None) for k in self.ADOPTER_SPECIFIC_KEYS}
        )
        sk_df['adopter_specific'] = sk_df['adopter_specific'].astype(str)
        sk_df['dcc'] = sk_df['dcc'].apply(lambda x: cast_to_list(x)).astype(object)
        sk_df['scope'] = sk_df['scope'].apply(lambda x: cast_to_list(x))
        sk_df['mhscope'] = (
            sk_df['mhscope'].apply(lambda x: cast_to_list(x)).astype(object)
        )

        if 'semver_tags' in sk_df.columns:
            sk_df['semver_tags'] = sk_df['semver_tags'].apply(lambda x: cast_to_list(x))

        if 'altver_tags' in sk_df.columns:
            sk_df['altver_tags'] = sk_df['altver_tags'].apply(lambda x: cast_to_list(x))

        if 'mtm_tags' in sk_df.columns:
            sk_df['mtm_tags'] = sk_df['mtm_tags'].apply(lambda x: cast_to_list(x))

        return sk_df

    def correct_columns_to_json(self, data: pd.DataFrame):
        log.info("Running correct_columns_to_json")
        if "internal_only" in data.columns:
            data = data.drop("internal_only", axis=1)

        data.fillna("", inplace=True)

        adopter_specific_df = (
            data['adopter_specific']
            .apply(lambda s: dict(ast.literal_eval(s)))
            .apply(pd.Series)
        )

        log.debug("done adopter_specific")
        adopter_specific_df = adopter_specific_df[
            ["ibmdocstype", "ibmdocsproduct", "ibmdocskey"]
        ]

        # we want to take ibmdocsproduct from the adopter_specific fields, so drop the upper level if it's there
        if 'ibmdocsproduct' in data.columns:
            data.drop(columns=["ibmdocsproduct"], inplace=True)
        data = pd.concat([data, adopter_specific_df], axis=1)

        # need to keep our metadata keys to exactly whats defined in the schema
        if 'adopter_specific' in data.columns:
            data.drop(columns=["adopter_specific"], inplace=True)

        is_raw = False
        if 'raw_body' in data.columns:
            data.rename(columns={"raw_body": "content"}, inplace=True)
            is_raw = True
        else:
            data.rename(columns={"body": "content"}, inplace=True)

        data.rename(
            columns={
                "dcdate": "publish_date",
                "scope": "scopes",
                "mhscope": "sub_scopes",
                "dcc": "digital_content_codes",
                "entitled": "is_entitled",
            },
            inplace=True,
        )

        data["is_public_ibmentitlement"] = data["ibmentitlement"].str.contains(
            "public", regex=False, na=True
        )
        data["is_essuperuser_ibmentitlement"] = data["ibmentitlement"].str.contains(
            "essuperuser", regex=False, na=True
        )
        data = self.entitlement_types_handler(data)
        data.drop(columns=["ibmentitlement"], inplace=True)

        data["publish_date"] = data['publish_date'].apply(lambda x: convert_datetime(x))

        data["last_updated"] = int(time.time_ns() / 1000000)

        data["title"] = data["title"].str.slice(0, 3000)
        data["description"] = data["description"].str.slice(0, 3000)
        data["url"] = data["url"].str.slice(0, 2000)
        data["language"] = data["language"].str.slice(0, 5)
        data["ibmdocstype"] = data["ibmdocstype"].str.slice(0, 20)
        data["ibmdocstype"] = data["ibmdocstype"].fillna("")
        data["ibmdocsproduct"] = data["ibmdocsproduct"].fillna("")
        data["ibmdocsproduct"] = data["ibmdocsproduct"].str.slice(0, 200)

        data["ibmdocskey"] = data["ibmdocskey"].astype(str)
        data["ibmdocskey"] = data["ibmdocskey"].str.slice(0, 50)
        data["field_keyword_01"] = data["field_keyword_01"].str.slice(0, 100)

        data["scopes"] = data["scopes"].apply(lambda x: truncate_list(x, 10))
        data["scopes"] = data["scopes"].apply(lambda x: truncate_list_elements(x, 40))

        data["sub_scopes"] = data["sub_scopes"].apply(lambda x: truncate_list(x, 20))

        data["sub_scopes"] = data["sub_scopes"].apply(
            lambda x: truncate_list_elements(x, 40)
        )

        data["digital_content_codes"] = data["digital_content_codes"].apply(
            lambda x: truncate_list(x, 20)
        )
        data["digital_content_codes"] = data["digital_content_codes"].apply(
            lambda x: truncate_list_elements(x, 20)
        )

        if 'semver_tags' in data.columns:
            data["semver_tags"] = data["semver_tags"].apply(
                lambda x: truncate_list(x, 20)
            )
            data["semver_tags"] = data["semver_tags"].apply(
                lambda x: truncate_list_elements(x, 20)
            )

        if 'altver_tags' in data.columns:
            data["altver_tags"] = data["altver_tags"].apply(
                lambda x: truncate_list(x, 20)
            )
            data["altver_tags"] = data["altver_tags"].apply(
                lambda x: truncate_list_elements(x, 20)
            )

        if 'mtm_tags' in data.columns:
            data["mtm_tags"] = data["mtm_tags"].apply(lambda x: truncate_list(x, 20))
            data["mtm_tags"] = data["mtm_tags"].apply(
                lambda x: truncate_list_elements(x, 20)
            )

        if is_raw:
            self.perform_table_extract(data)

        keys = list(data.columns)
        ret = data.to_dict(orient='records')  # 2x content somewhere
        log.info("Done correct_columns_to_json")

        return ret, keys

    def load_data(self, embeddings, metadata, page_content, collection, chunk_type):
        start_time = time.time()
        data = self.get_data_fields(embeddings, metadata, page_content)

        collection.insert(data)
        collection.load()
        collection.flush()
        execution_time = time.time() - start_time
        log.info(
            "Successfully loaded {} embeddings with '{}' as chunk_type in {} secs".format(
                len(embeddings), chunk_type, execution_time
            )
        )

    def entitlement_types_handler(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def get_data_fields(self, embeddings, metadata, page_content):
        data = [
            [metadata[i]['chunk_num'] for i in range(len(embeddings))],  # chunk_num
            [metadata[i]["title"] for i in range(len(metadata))],
            [metadata[i]["description"] for i in range(len(metadata))],
            [metadata[i]["url"] for i in range(len(metadata))],
            [metadata[i]["language"] for i in range(len(metadata))],
            [metadata[i]["last_updated"] for i in range(len(metadata))],
            [metadata[i]["is_entitled"] for i in range(len(metadata))],
            [metadata[i]["is_public_ibmentitlement"] for i in range(len(metadata))],
            [
                metadata[i]["is_essuperuser_ibmentitlement"]
                for i in range(len(metadata))
            ],
            [metadata[i]["publish_date"] for i in range(len(metadata))],
            [metadata[i]["ibmdocstype"] for i in range(len(metadata))],
            [metadata[i]["ibmdocsproduct"] for i in range(len(metadata))],
            [metadata[i]["scopes"] for i in range(len(metadata))],
            [metadata[i]["sub_scopes"] for i in range(len(metadata))],
            [metadata[i]["digital_content_codes"] for i in range(len(metadata))],
            [metadata[i]["field_keyword_01"] for i in range(len(metadata))],
            [metadata[i]["ibmdocskey"] for i in range(len(metadata))],
            [metadata[i]["latest_version"] for i in range(len(metadata))],
            [metadata[i]["semver_tags"] for i in range(len(metadata))],
            [metadata[i]["altver_tags"] for i in range(len(metadata))],
            [metadata[i]["mtm_tags"] for i in range(len(metadata))],
            [metadata[i]["is_table"] for i in range(len(metadata))],
            page_content,
            embeddings,
        ]
        return data

    def get_schema_fields(self, fields_dict: dict, dim: int) -> list[FieldSchema]:
        fields = [
            FieldSchema(
                name="doc_id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
                description=fields_dict.get('doc_id', ''),
            ),
            FieldSchema(
                name="chunk_num",
                dtype=DataType.INT16,
                description=fields_dict.get('chunk_num', ''),
            ),
            FieldSchema(
                name="title",
                dtype=DataType.VARCHAR,
                max_length=12000,
                description=fields_dict.get('title', ''),
            ),
            FieldSchema(
                name="description",
                dtype=DataType.VARCHAR,
                max_length=12000,
                description=fields_dict.get('description', ''),
            ),
            FieldSchema(
                name="url",
                dtype=DataType.VARCHAR,
                max_length=8000,
                description=fields_dict.get('url', ''),
            ),
            FieldSchema(
                name="language",
                dtype=DataType.VARCHAR,
                max_length=20,
                description=fields_dict.get('language', ''),
            ),
            FieldSchema(
                name="last_updated",
                dtype=DataType.INT64,
                description=fields_dict.get('last_updated', ''),
            ),
            FieldSchema(
                name="is_entitled",
                dtype=DataType.BOOL,
                description=fields_dict.get('is_entitled', ''),
            ),
            FieldSchema(
                name="is_public_ibmentitlement",
                dtype=DataType.BOOL,
                description=fields_dict.get('is_public_ibmentitlement', ''),
            ),
            FieldSchema(
                name="is_essuperuser_ibmentitlement",
                dtype=DataType.BOOL,
                description=fields_dict.get('is_essuperuser_ibmentitlement', ''),
            ),
            FieldSchema(
                name="publish_date",
                dtype=DataType.INT64,
                description=fields_dict.get('publish_date', ''),
            ),
            FieldSchema(
                name="ibmdocstype",
                dtype=DataType.VARCHAR,
                max_length=80,
                description=fields_dict.get('ibmdocstype', ''),
            ),
            FieldSchema(
                name="ibmdocsproduct",
                dtype=DataType.VARCHAR,
                max_length=800,
                description=fields_dict.get('ibmdocsproduct', ''),
            ),
            FieldSchema(
                name="scopes",
                dtype=DataType.ARRAY,
                element_type=DataType.VARCHAR,
                max_capacity=10,
                max_length=160,
                description=fields_dict.get('scopes', ''),
            ),
            FieldSchema(
                name="sub_scopes",
                dtype=DataType.ARRAY,
                element_type=DataType.VARCHAR,
                max_capacity=20,
                max_length=160,
                description=fields_dict.get('sub_scopes', ''),
            ),
            FieldSchema(
                name="digital_content_codes",
                dtype=DataType.ARRAY,
                element_type=DataType.VARCHAR,
                max_capacity=20,
                max_length=80,
                description=fields_dict.get('digital_content_codes', ''),
            ),
            FieldSchema(
                name="field_keyword_01",
                dtype=DataType.VARCHAR,
                max_length=400,
                description=fields_dict.get('field_keyword_01', ''),
            ),
            FieldSchema(
                name="ibmdocskey",
                dtype=DataType.VARCHAR,
                max_length=200,
                description=fields_dict.get('ibmdocskey', ''),
            ),
            FieldSchema(
                name="latest_version",
                dtype=DataType.BOOL,
                description=fields_dict.get('latest_version', ''),
            ),
            FieldSchema(
                name="semver_tags",
                dtype=DataType.ARRAY,
                element_type=DataType.VARCHAR,
                max_capacity=20,
                max_length=80,
                description=fields_dict.get('semver_tags', ''),
            ),
            FieldSchema(
                name="altver_tags",
                dtype=DataType.ARRAY,
                element_type=DataType.VARCHAR,
                max_capacity=20,
                max_length=80,
                description=fields_dict.get('altver_tags', ''),
            ),
            FieldSchema(
                name="mtm_tags",
                dtype=DataType.ARRAY,
                element_type=DataType.VARCHAR,
                max_capacity=20,
                max_length=80,
                description=fields_dict.get('mtm_tags', ''),
            ),
            FieldSchema(
                name="is_table",
                dtype=DataType.BOOL,
                description=fields_dict.get('is_table', ''),
            ),
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=16000,
                description=fields_dict.get('content', ''),
            ),
            FieldSchema(
                name="doc_vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=dim,
                description=fields_dict.get('doc_vector', ''),
            ),
        ]
        return fields

    def run(self):
        data, clean_metadata_keys = self.correct_columns_to_json(
            self.elastic_cos_pull()
        )
        return data, clean_metadata_keys
    
   