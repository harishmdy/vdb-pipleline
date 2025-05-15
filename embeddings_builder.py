import logging as logger
import os

import numpy as np
import pandas as pd
from langchain.schema.document import Document

# from ibm.unifiedsearchvectors.data_builder.base import MitaDataBuilder
from ibm.unifiedsearchvectors.utils.connect import CosManager

log = logger.getLogger("ibmsearch")
DOC_BATCH_SIZE = int(os.getenv('DOC_BATCH_SIZE', 1000))


class EmbeddingsBuilder:
    def __init__(self, data: list[dict]):
        # super().__init__(data)
        self.data: list[dict] = data

    def clean(self) -> None:
        # [insert cleaning code here]
        return

    def remove_missing(
        self,
        data: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        '''
        data: data in json format, must have all keys in REQD_COLS
        Return:
            Data that has all of the required columns
            Data that does not have all of the required columns
        '''
        REQD_COLS = {'title', 'description', 'content', 'url'}
        return [
            d
            for d in data
            if len(REQD_COLS.intersection(set(d.keys()))) == len(REQD_COLS)
        ], [
            d
            for d in data
            if len(REQD_COLS.intersection(set(d.keys()))) != len(REQD_COLS)
        ]

    # Maybe one of two, body or description
    def get_documents(
        self,
        data: list[dict],
        metadata_keys: list[str],
        content_keys: list[str],
        #    content_keys: list[str] = [SRC_TITLE_COL, SRC_DESC_COL, SRC_BODY_COL],
        content_sep: str = ' ',
    ) -> list[Document]:
        new_data, _ = self.remove_missing(data)
        ret_list = []
        content_missing_count = 0

        for doc in new_data:
            metadata_dict = {"Title": None}  # Need this for HTML splitter to work
            metadata_dict["is_table"] = False
            for k in metadata_keys:
                if k in doc.keys() and k != "tables" and k != "is_table":
                    metadata_dict[k] = doc[k]
                else:
                    metadata_dict[k] = ""
            content_key_checker = True
            for key in content_keys:
                if doc[key] is None:
                    content_missing_count += 1
                    content_key_checker = False

            if content_key_checker:
                ret_list.append(
                    Document(
                        metadata=metadata_dict,
                        page_content=content_sep.join(doc[key] for key in content_keys),
                    )
                )
                if "tables" in doc.keys():
                    table_metadata_dict = metadata_dict.copy()
                    table_metadata_dict["is_table"] = True
                    if doc["tables"] is not None:
                        for table_dict in doc["tables"]:
                            table_metadata_dict["Descriptive Sentences"] = table_dict[
                                "Description"
                            ]
                            table_metadata_dict["Caption"] = table_dict["Caption"]
                            table_dict["Table"] = (
                                table_metadata_dict["Descriptive Sentences"]
                                + "\n"
                                + table_metadata_dict["Caption"]
                                + "\n"
                                + table_dict["Table"]
                            )

                            ret_list.append(
                                Document(
                                    metadata=table_metadata_dict,
                                    page_content=table_dict["Table"],
                                )
                            )

        log.info(f"content none count: {content_missing_count}")
        return ret_list

    # def split_docs(self, documents: list[Document], batch_size: int):
    #     split_size = max(len(documents) // batch_size, 1)
    #     batched_docs = np.array_split(documents, split_size)
    #     return batched_docs
    def split_docs(self, documents: list[Document], batch_size: int):
        """
        Split documents into batches of size batch_size.
        We want to keep documents with same URL together,
        so we find splitting indices that keep documents of same URL together

        Parameters:
        documents (list[Document]): List of documents to be split.
        batch_size (int): Size of each batch.

        Returns:
        list[list[Document]]: List of batches, where each batch is a list of documents.
        """
        original_split_indices = [
            i for i in range(batch_size, len(documents), batch_size)
        ]
        avoiding_split = False
        new_split_indices = []
        for i in range(len(documents)):
            if i in original_split_indices or avoiding_split:
                prev_url = documents[i - 1].metadata["url"]
                avoiding_split = True
                if prev_url != documents[i].metadata["url"]:
                    new_split_indices.append(i)
                    avoiding_split = False
        return np.split(documents, new_split_indices)

    def pre_transform_validate(self) -> None:
        """Run checks for data correctness pre-transformation."""
        # [insert checks here]

    def transform(self) -> None:
        # [insert transforming code here]
        return

    def post_transform_validate(self) -> None:
        """Run checks for data correctness post-transformation."""
        # [insert checks here]

    def run(self, metadata_keys: list[str], content_keys: list[str]) -> list[Document]:
        self.pre_transform_validate()
        self.clean()
        self.transform()
        self.post_transform_validate()
        documents = self.get_documents(
            data=self.data,
            metadata_keys=metadata_keys,
            content_keys=content_keys,
        )
        # first_100 = documents[0:100]
        batched_docs = self.split_docs(documents, batch_size=DOC_BATCH_SIZE)
        log.info(
            f"split {len(documents)} unchunked documents into {len(batched_docs)} using {DOC_BATCH_SIZE}"
        )
        del self.data
        return batched_docs

    @classmethod
    def from_bucket(cls, cos_manager: CosManager) -> 'ExampleBuilder':
        # Run COS aggregations, like so:
        # regex_patterns = [f"file_{i}.csv" for i in range(10)]
        # data = cos_manager.aggregate_files(regex_patterns)

        example_data = pd.DataFrame()
        return cls(example_data)