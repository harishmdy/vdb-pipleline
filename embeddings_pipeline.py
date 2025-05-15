import ast
import json
import logging as logger
import math
import multiprocessing as mp
import os
import os.path
import re
import sys
import time
from functools import partial
from itertools import chain
from typing import cast

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from langchain.schema.document import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from pydantic import BaseModel
from pymilvus import (
    BulkInsertState,
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from sentence_transformers import SentenceTransformer

# from carrot.pipeline import DataPipelineBase
from ibm.unifiedsearchvectors.data_builder.cloud_docs import CloudDocs
from ibm.unifiedsearchvectors.data_builder.cloud_docs_int import (
    CloudDocsInt,
)
from ibm.unifiedsearchvectors.data_builder.embeddings_builder import EmbeddingsBuilder
from ibm.unifiedsearchvectors.data_builder.general_technical_docs import (
    GeneralTechnicalDocs,
)
from ibm.unifiedsearchvectors.data_builder.ibm_design import IbmDesign
from ibm.unifiedsearchvectors.data_builder.ibm_docs import IbmDocs
from ibm.unifiedsearchvectors.data_builder.ibm_docs_ent import (
    IbmDocsEnt,  # added for entitlement
)
from ibm.unifiedsearchvectors.data_builder.marketing import Marketing
from ibm.unifiedsearchvectors.data_builder.partner_plus_ent import PartnerPlusEnt
from ibm.unifiedsearchvectors.data_builder.redhat import RedHat
from ibm.unifiedsearchvectors.data_builder.sales_announcements import SalesAnnouncements
from ibm.unifiedsearchvectors.data_builder.support import Support
from ibm.unifiedsearchvectors.data_builder.support_ent import (
    SupportEnt,  # added for entitlement
)
from ibm.unifiedsearchvectors.data_builder.support_int import (
    SupportInt,  # added for entitlement
)
from ibm.unifiedsearchvectors.data_builder.training import Training
from ibm.unifiedsearchvectors.utils.connect import CosManager
from ibm.unifiedsearchvectors.utils.const import EXAMPLE_COS_BUCKET
from ibm.unifiedsearchvectors.utils.functions import (  # ENCODE_DEVICE,
    check_alias,
    configure_logging,
    connect_cos,
    connect_milvus,
    connect_milvus_cos,
    create_alias,
    format_json,
    wait_tasks,
)
from ibm.unifiedsearchvectors.utils.MITAHTMLSectionSplitter import (
    MITAHTMLSectionSplitter,
)

log = logger.getLogger("ibmsearch")

# MODEL_NAME = "thenlper/gte-small"
MODEL_PARAMS = {"split_type": "recursive", "chunk_size": 2000, "chunk_overlap": 200}
# start_batch_num = 0

ENCODE_DEVICE = os.getenv('ENCODE_DEVICE', None)
EMBED_API_KEY = os.getenv('EMBED_API_KEY', None)
EMBED_API_URL_BASE = os.getenv('EMBED_API_URL_BASE', None)
PROCESS_POOL_COUNT = int(os.getenv('PROCESS_POOL_COUNT', 1))

# MODEL_NAME = os.getenv('MODEL_NAME', 'ibm/slate.30m.english.rtrvr')
MODEL_NAME = os.getenv('MODEL_NAME', None)
MODEL_PATH = os.getenv('MODEL_PATH', "..")

CHECK_BATCH_PROPORTION = float(os.getenv('CHECK_BATCH_PROPORTION', "0.20"))


class EmbeddingsConfig(BaseModel):
    start_batch_num: int
    doc_type: str
    collection_name: str
    pull_elastic: str
    is_html: bool
    bulk_api: int


class EmbeddingsPipeline:
    def __init__(self, config: dict):
        super().__init__()
        configure_logging()
        self.config = EmbeddingsConfig.model_validate(config)
        self.embed_pool = None
        self.process_pool = None
        log.info(f"Starting pipeline with {config}")

    def build(self) -> None:
        """Instantiate builders."""
        # -- Just placeholders for inputs, remove these --
        self.cos: CosManager
        # self.embeddings_builder = EmbeddingsBuilder({"url": "test"})
        self.bucket = EXAMPLE_COS_BUCKET
        # --

        # This is more what your build() should look like:
        # cos = CosManager.create_from_config(**EXAMPLE_COS_BUCKET)
        # self.example_builder = ExampleBuilder.from_bucket(cos)

    def remove_empty_chunks(self, split_docs: list[Document]) -> list[Document]:
        for i in range(len(split_docs) - 1, -1, -1):
            # if i%2==0:
            #     del(split_docs[i])

            if ('is_table' in split_docs[i].metadata) & (
                split_docs[i].metadata['is_table']
            ):
                pass
            else:
                already_deleted = False
                clean_page_content = (
                    split_docs[i].page_content.replace("\n", "").replace(" ", "")
                )

                if (
                    len(split_docs[i].page_content) < 150
                ):  # remove all chunks less than 150 chars
                    log.debug(f"deleting from url: {split_docs[i].metadata['url']}")
                    del split_docs[i]
                    already_deleted = True

                if clean_page_content == '' and not already_deleted:
                    del split_docs[i]
                    already_deleted = True

                if not already_deleted:
                    title = split_docs[i].metadata.get('title')
                    h1 = split_docs[i].metadata.get('h1')
                    if h1 is not None and title is not None:
                        clean_title = title.replace("\n", "").replace(" ", "")
                        clean_h1 = h1.replace("\n", "").replace(" ", "")
                        if (
                            clean_title == clean_page_content
                            or clean_h1 == clean_page_content
                        ):
                            del split_docs[i]
                    elif title is not None:
                        clean_title = title.replace("\n", "").replace(" ", "")
                        if clean_title == clean_page_content:
                            del split_docs[i]
        return split_docs

    def pre_pend_chunks(self, split_docs: list[Document]) -> list[Document]:
        """
        Prepend title, descriptive sentences, and caption to table chunks
        Prepend the title and header 1 to the page content
        Only prepend when things are not already present

        Args:
            split_docs (list[Document]): A list of Document objects.

        Returns:
            list[Document]: The updated list of Document objects with prepended title and header 1.
        """

        for split_doc in split_docs:
            title = self.appendIfNotEmpty(split_doc.metadata.get('title', ''), '\n')

            if ('is_table' in split_doc.metadata) and (split_doc.metadata['is_table']):
                desc = self.appendIfNotEmpty(
                    split_doc.metadata.get('Descriptive Sentences', ''), '\n'
                )
                caption = self.appendIfNotEmpty(
                    split_doc.metadata.get('Caption', ''), '\n'
                )

                # clear these out now
                split_doc.metadata.pop('Descriptive Sentences', None)
                split_doc.metadata.pop('Caption', None)

                clean_desc = desc.replace("\n", "").replace(" ", "")
                clean_caption = caption.replace("\n", "").replace(" ", "")
                clean_table_content = split_doc.page_content.replace("\n", "").replace(
                    " ", ""
                )
                if clean_caption in clean_desc[0 : len(clean_caption)]:
                    if clean_desc != clean_table_content[0 : len(clean_desc)]:
                        split_doc.page_content = desc + split_doc.page_content
                elif clean_desc in clean_caption:
                    if clean_caption != clean_table_content[0 : len(clean_caption)]:
                        split_doc.page_content = caption + split_doc.page_content
                else:
                    split_doc.page_content = caption + desc + split_doc.page_content
                split_doc.page_content = title + split_doc.page_content

            else:

                # h1 = self.appendIfNotEmpty(
                #         split_doc.metadata.get('Header 1', ''), '\n'
                #     )

                # split_doc.page_content = title + h1 + split_doc.page_content
                h1 = split_doc.metadata.get('Header 1', '')
                if (
                    len(split_doc.page_content) > 250
                ):  # Only prepend if greater than 250 chars
                    if h1 is not None and title is not None:
                        clean_title = title.replace("\n", "").replace(" ", "")
                        clean_h1 = h1.replace("\n", "").replace(" ", "")
                        clean_page_content = split_doc.page_content.replace(
                            "\n", ""
                        ).replace(" ", "")
                        if (
                            clean_title != clean_page_content
                            and clean_page_content != ""
                        ):
                            if clean_title in clean_h1:
                                if clean_h1 != clean_page_content[0 : len(clean_h1)]:
                                    split_doc.page_content = (
                                        h1 + "\n" if h1 else ""
                                    ) + split_doc.page_content
                            elif clean_h1 in clean_title:
                                if (
                                    clean_title
                                    != clean_page_content[0 : len(clean_title)]
                                ):
                                    split_doc.page_content = (
                                        title + split_doc.page_content
                                    )
                            else:
                                if (
                                    clean_title
                                    != clean_page_content[0 : len(clean_title)]
                                    and clean_h1
                                    != clean_page_content[0 : len(clean_h1)]
                                ):
                                    split_doc.page_content = (
                                        title
                                        + (h1 + "\n" if h1 else "")
                                        + split_doc.page_content
                                    )
                                elif (
                                    clean_title
                                    != clean_page_content[0 : len(clean_title)]
                                ):
                                    split_doc.page_content = (
                                        title + split_doc.page_content
                                    )
                                elif clean_h1 != clean_page_content[0 : len(clean_h1)]:
                                    split_doc.page_content = h1 + split_doc.page_content
                    elif title is not None:

                        clean_title = title.replace("\n", "").replace(" ", "")
                        clean_page_content = split_doc.page_content.replace(
                            "\n", ""
                        ).replace(" ", "")
                        if (
                            clean_title != clean_page_content
                            and clean_page_content != ""
                        ):
                            if clean_title != clean_page_content[0 : len(clean_title)]:
                                split_doc.page_content = title + split_doc.page_content

        return split_docs

    def appendIfNotEmpty(self, text: str, append_text: str) -> str:
        if (
            (text is not None)
            and (not (text.strip() == ""))
            and (append_text is not None)
        ):
            return text + append_text
        else:
            return text

    def generate_embeddings(
        self,
        split_docs: list[Document],
        model: SentenceTransformer,
        batch_size: int = 64,
    ) -> tuple[np.ndarray, list, list]:
        '''
        docs: list of Document objects with title, description, body, and URL metadata
        model_name: name of HuggingFace model used
        model: HuggingFace model instance
        split_type: chunking method used, either "fixed" or "recursive"
        tokens_per_chunk: how many tokens in each chunk
        chunk_overlap: number of overlapping tokens in each chunk
        batch_size: batch size used for embedding creation
        Return: list of embeddings by chunk grouped by document
                parallel list of metadata of each document
        '''
        start_time = time.time()
        embeddings = None
        if (ENCODE_DEVICE is not None) and (ENCODE_DEVICE == 'api'):
            embeddings = cast(np.ndarray, self.embed_api(split_docs))
        elif self.embed_pool is not None:
            embeddings = model.encode_multi_process(
                [d.page_content for d in split_docs], self.embed_pool
            )
        else:
            embeddings = cast(
                np.ndarray,
                model.encode(
                    [d.page_content for d in split_docs],
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    device=ENCODE_DEVICE,
                ),
            )
        metadata = [d.metadata for d in split_docs]
        # Truncating page content here to be no more than 4000 characters, make sure it fits in schema
        page_content = [d.page_content[0:4000] for d in split_docs]

        execution_time = time.time() - start_time
        log.info(f'Embedding execution time in seconds: {execution_time:.2f}')
        return embeddings, metadata, page_content

    def html_chunk_docs(self, html_splitter: MITAHTMLSectionSplitter, docs):
        split_docs = html_splitter.split_documents(docs)
        # split_docs = splitter.split_documents(html_split_docs)  # type: ignore
        split_docs = self.remove_empty_chunks(split_docs)
        split_docs = self.pre_pend_chunks(split_docs)
        for doc in split_docs:
            # doc.page_content = re.sub(
            # r"\n{3,}", "\n" * 2, doc.page_content)
            pattern = re.compile(r'(\s*\n\s*){3,}')
            doc.page_content = pattern.sub('\n', doc.page_content)
        return split_docs

    def html_chunk_doc(self, html_splitter: MITAHTMLSectionSplitter, doc: Document):
        return self.html_chunk_docs(html_splitter, [doc])

    def split_and_embed(
        self,
        docs: list[Document],
        model_name: str,
        model: SentenceTransformer,
        split_type: str = "recursive",
        chunk_size: int = 250,
        chunk_overlap: int = 10,
        batch_size: int = 64,
        schema_fields_set: set = None,
    ) -> tuple[np.ndarray, list, list]:
        '''
        docs: list of Document objects with title, description, body, and URL metadata
        model_name: name of HuggingFace model used
        model: HuggingFace model instance
        split_type: chunking method used, either "fixed" or "recursive"
        tokens_per_chunk: how many tokens in each chunk
        chunk_overlap: number of overlapping tokens in each chunk
        batch_size: batch size used for embedding creation
        Return: list of embeddings by chunk grouped by document
                parallel list of metadata of each document
        '''
        if not split_type.startswith("fixed") and not split_type.startswith(
            "recursive"
        ):
            raise ValueError("Unsupported split type")
        if split_type.startswith("fixed"):
            splitter = SentenceTransformersTokenTextSplitter(
                model_name=model_name,
                tokens_per_chunk=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        if split_type.startswith("recursive"):
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        if self.config.is_html:
            headers_to_split_on = [("h1", "Header 1")]
            # TODO: Maybe fix tables being ran through the html splitter

            html_splitter = MITAHTMLSectionSplitter(
                headers_to_split_on,
                xslt_path="src/ibm/unifiedsearchvectors/resources/html_splitter.xslt",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            if self.process_pool is not None:
                func = partial(self.html_chunk_doc, html_splitter)
                split_docs = list(
                    chain.from_iterable(self.process_pool.map(func, docs))
                )
            else:
                split_docs = self.html_chunk_docs(html_splitter, docs)

            # remove extra metadata fields since it breaks the bulk load

        else:
            split_docs = splitter.split_documents(docs)  # type: ignore

        # clean out left over meta data that's not in the set of schema columns for db
        fields = None
        if schema_fields_set is not None:
            for doc in split_docs:
                fields = fields or list(doc.metadata.keys())
                for field in fields:
                    if field not in schema_fields_set:
                        doc.metadata.pop(field, None)
        embeddings, metadata, page_content = self.generate_embeddings(
            split_docs, model, batch_size
        )
        return embeddings, metadata, page_content

    def add_chunk_number(self, metadata: list):
        count_dict = {}
        for i in range(len(metadata)):
            # metadata[i] = {key: metadata[i][key] for key in docs_obj.METADATA_KEYS}
            url = metadata[i]["url"]
            if url in count_dict:
                count_dict[url] += 1
            else:
                count_dict[url] = 1
            metadata[i]["chunk_num"] = np.int16(count_dict[url])

    def embed_api(self, split_docs: list):
        string_array = [d.page_content for d in split_docs]

        headers = {'Authorization': f'Bearer {EMBED_API_KEY}'}
        data = {
            "inputs": string_array,
            "embed_normalization": "False",
        }

        url = str(EMBED_API_URL_BASE) + os.getenv('MODEL_NAME')
        log.debug(f"requests for {len(split_docs)} embeddings")
        response = requests.post(url, json=data, headers=headers)
        if response.status_code != 200:
            reason = response.reason.replace('\n', ' ').replace(
                '\r', ' '
            )  # NEUTRALIZATION
            response_str = (
                str(response).replace('\n', ' ').replace('\r', ' ')
            )  # NEUTRALIZATION

            log.error(
                f"{response.status_code} - {reason} response from embed api: {response_str} "
            )

        return response.json()

    def get_model(self):
        if MODEL_NAME == "thenlper/gte-small":
            model = SentenceTransformer(
                MODEL_NAME,
                revision="d333cac34d43ccb09cb9d97b2d558b2c4315558f",  # pragma: allowlist secret
            )
        elif MODEL_NAME.startswith("ibm"):
            if not os.path.exists(MODEL_PATH + "/slate.30m.english.rtrvr"):
                log.info(
                    f"The model, slate.30m.english.rtrvr is downloading into {MODEL_PATH}."
                )
                obj_config = connect_cos()
                obj_config.download_folder_from_cos("dmf_models/", MODEL_PATH)
            else:
                log.info(
                    f"The model, slate.30m.english.rtrvr is already exists in  {MODEL_PATH}."
                )
            model = SentenceTransformer(MODEL_PATH + "/slate.30m.english.rtrvr")
        else:
            model = SentenceTransformer(MODEL_NAME)
        return model

    def do_bulk_load_for_batch(
        self,
        collection: Collection,
        task_id_dict: dict,
        data_json: dict,
        file_name: str,
        batch_num: int,
        total_batches: int,
        check_batch_size: int,
    ):

        num_embeddings = len(data_json["rows"])

        obj_config = connect_milvus_cos()
        log.info(
            f"Uploading batch {self.config.start_batch_num} out of {total_batches - 1} to {file_name}. {num_embeddings} embeddings uploaded!"
        )
        obj_config.insert_json(data_json, file_name)
        task_id = utility.do_bulk_insert(
            collection_name=collection.name, files=[file_name], timeout=300
        )
        log.info(
            "Task id for batch number %d: %s", self.config.start_batch_num, task_id
        )

        task_id_dict[task_id] = (self.config.start_batch_num, num_embeddings)
        self.config.start_batch_num += 1

        if (batch_num + 1) % check_batch_size == 0:
            log.info(
                f"Checking task_ids for batches {batch_num + 1 - check_batch_size} through {batch_num}..."
            )
            states = wait_tasks(
                task_id_dict,
                BulkInsertState.ImportCompleted,
                log_completion=True,
                timeout=300,
                retry_limit=3,
            )
            task_id_dict.clear()
        elif ((batch_num + 1) == total_batches) and (
            (batch_num + 1) % check_batch_size != 0
        ):
            log.info(
                f"Checking task_ids for batches {check_batch_size*((batch_num + 1) // check_batch_size)} through {total_batches - 1}"
            )
            states = wait_tasks(
                task_id_dict,
                BulkInsertState.ImportCompleted,
                log_completion=True,
                timeout=300,
                retry_limit=3,
            )

        collection.load()
        collection.flush()

    def final_load(
        self, collection: Collection, batched_docs: list[list[Document]], docs_obj
    ) -> None:
        log.info(
            f"Starting embedding generation to milvus on {len(batched_docs)} batches of docs"
        )
        batched_docs = batched_docs[self.config.start_batch_num :]
        collection.load()

        # Add logic for different models based on config arg here
        model = self.get_model()
        log.info(f"ENCODE_DEVICE: {ENCODE_DEVICE}")
        if ENCODE_DEVICE is not None and ENCODE_DEVICE.startswith('['):
            log.info("Creating embedding process pool")
            encode_devices: list = ast.literal_eval(ENCODE_DEVICE)
            self.embed_pool = model.start_multi_process_pool(
                target_devices=encode_devices
            )
            log.info(f"Embedding process pool created with {str(encode_devices)}")

        chunk_size = MODEL_PARAMS["chunk_size"]
        chunk_overlap = MODEL_PARAMS["chunk_overlap"]

        if PROCESS_POOL_COUNT > 1:
            self.process_pool = mp.Pool(PROCESS_POOL_COUNT)
        for batch in batched_docs:
            embeddings, metadata, page_content = self.split_and_embed(
                docs=batch,
                model_name=MODEL_NAME,
                model=model,
                split_type=MODEL_PARAMS["split_type"],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            self.add_chunk_number(metadata)

            log.debug(f"Number of embeddings loading to Milvus: {len(embeddings)}")

            docs_obj.load_data(
                embeddings,
                metadata,
                page_content,
                collection,
                MODEL_PARAMS["split_type"],
            )

            log.info(
                f"Batch number {self.config.start_batch_num} of {len(batched_docs)} total batches and {len(embeddings)} chunks done!!!"
            )
            self.config.start_batch_num += 1

        # close the pool if we created one
        if self.embed_pool is not None:
            model.stop_multi_process_pool(self.embed_pool)
        if self.process_pool is not None:
            self.process_pool.close()

    def final_bulk_load(self, collection, batched_docs, docs_obj) -> None:
        log.info(
            f"Starting embedding generation to Milvus on {len(batched_docs)} batches of docs"
        )
        batched_docs = batched_docs[self.config.start_batch_num :]
        collection.load()

        log.info(f"ENCODE_DEVICE: {ENCODE_DEVICE}")
        model = self.get_model()
        chunk_size = MODEL_PARAMS["chunk_size"]
        chunk_overlap = MODEL_PARAMS["chunk_overlap"]

        CHECK_BATCH_SIZE = math.ceil(len(batched_docs) * CHECK_BATCH_PROPORTION)
        log.info(f"Checking task states for every {CHECK_BATCH_SIZE} batches")
        task_id_dict = {}

        schema_fields_set = set([x.name for x in docs_obj.get_schema_fields({}, 0)])

        for batch_num, batch in enumerate(batched_docs):

            # Generate embeddings and prepare data
            embeddings, metadata, page_content = self.split_and_embed(
                docs=batch,
                model_name=MODEL_NAME,
                model=model,
                split_type=MODEL_PARAMS["split_type"],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                schema_fields_set=schema_fields_set,
            )
            self.add_chunk_number(metadata)

            # uncomment to test batch failure
            # adding a column that's not defined in the schema for this batch
            # if batch_num == 30:
            #     for i in range(len(metadata)):
            #         metadata[i]['test_dcc'] = 'TEST'

            file_name = f"{collection.name}/{docs_obj.DOCS_TYPE}_{str(self.config.start_batch_num)}.json"
            data_json = format_json(embeddings, metadata, page_content)
            self.do_bulk_load_for_batch(
                collection,
                task_id_dict,
                data_json,
                file_name,
                batch_num,
                len(batched_docs),
                CHECK_BATCH_SIZE,
            )
        log.info("All batches have been processed successfully.")

    def run(self) -> None:
        """Run builders and pipeline functions."""
        # Take as config arg?
        connect_milvus()

        if self.config.doc_type == IbmDocs.DOCS_TYPE:
            docs_obj = IbmDocs(pull_elastic=self.config.pull_elastic)
            content_keys = IbmDocs.CONTENT_KEYS
        elif self.config.doc_type == SalesAnnouncements.DOCS_TYPE:
            docs_obj = SalesAnnouncements(pull_elastic=self.config.pull_elastic)
            content_keys = SalesAnnouncements.CONTENT_KEYS
        elif self.config.doc_type == Support.DOCS_TYPE:
            docs_obj = Support(pull_elastic=self.config.pull_elastic)
            content_keys = Support.CONTENT_KEYS
        elif self.config.doc_type == CloudDocs.DOCS_TYPE:
            docs_obj = CloudDocs(pull_elastic=self.config.pull_elastic)
            content_keys = CloudDocs.CONTENT_KEYS
        elif self.config.doc_type == Marketing.DOCS_TYPE:
            docs_obj = Marketing(pull_elastic=self.config.pull_elastic)
            content_keys = Marketing.CONTENT_KEYS
        elif self.config.doc_type == GeneralTechnicalDocs.DOCS_TYPE:
            docs_obj = GeneralTechnicalDocs(pull_elastic=self.config.pull_elastic)
            content_keys = GeneralTechnicalDocs.CONTENT_KEYS
        elif (
            self.config.doc_type == IbmDocsEnt.DOCS_TYPE
        ):  # Added for IBM_DOCS Entitled
            docs_obj = IbmDocsEnt(pull_elastic=self.config.pull_elastic)
            content_keys = IbmDocsEnt.CONTENT_KEYS
        elif (
            self.config.doc_type == SupportEnt.DOCS_TYPE
        ):  # Added for Support_DOCS Entitled
            docs_obj = SupportEnt(pull_elastic=self.config.pull_elastic)
            content_keys = SupportEnt.CONTENT_KEYS
        elif (
            self.config.doc_type == SupportInt.DOCS_TYPE
        ):  # Added for Support_DOCS Internal
            docs_obj = SupportInt(pull_elastic=self.config.pull_elastic)
            content_keys = SupportInt.CONTENT_KEYS
        elif (
            self.config.doc_type == CloudDocsInt.DOCS_TYPE
        ):  # Added for Cloud_DOCS Internal
            docs_obj = CloudDocsInt(pull_elastic=self.config.pull_elastic)
            content_keys = CloudDocsInt.CONTENT_KEYS
        elif self.config.doc_type == RedHat.DOCS_TYPE:
            docs_obj = RedHat(pull_elastic=self.config.pull_elastic)
            content_keys = RedHat.CONTENT_KEYS
        elif self.config.doc_type == Training.DOCS_TYPE:
            docs_obj = Training(pull_elastic=self.config.pull_elastic)
            content_keys = Training.CONTENT_KEYS
        elif self.config.doc_type == PartnerPlusEnt.DOCS_TYPE:
            docs_obj = PartnerPlusEnt(pull_elastic=self.config.pull_elastic)
            content_keys = PartnerPlusEnt.CONTENT_KEYS
        elif self.config.doc_type == IbmDesign.DOCS_TYPE:
            docs_obj = IbmDesign(pull_elastic=self.config.pull_elastic)
            content_keys = IbmDesign.CONTENT_KEYS
        else:
            log.error("Unknown Type!!! " + self.config.doc_type)

        if self.config.pull_elastic == "only_pull":
            if self.config.doc_type == RedHat.DOCS_TYPE:
                docs_obj.graphql_cos_pull()
            else:
                docs_obj.elastic_cos_pull()
        elif (self.config.pull_elastic == "only_embed") or (
            self.config.pull_elastic == "both"
        ):
            create_alias(docs_obj=docs_obj, collection_name=self.config.collection_name)
            c_user, c_etl = check_alias(self.config.collection_name)

            doc_data, clean_metadata_keys = docs_obj.run()
            # del docs_obj.data
            # Doc objects are still around and have the original dataframe from elastic
            # plus we have the json documents here in doc_data.

            embeddings_builder = EmbeddingsBuilder(doc_data)
            metadata_keys = clean_metadata_keys
            del doc_data
            data = embeddings_builder.run(
                metadata_keys=metadata_keys, content_keys=content_keys
            )
            # del embeddings_builder.data

            c_etl_name = c_etl.name
            if self.config.start_batch_num == 0:
                utility.drop_collection(c_etl_name)
                docs_obj.create_collection(collection_name=c_etl_name, dim=384)
                c_etl = Collection(c_etl_name)

            if self.config.bulk_api == 1:
                self.final_bulk_load(
                    batched_docs=data, collection=c_etl, docs_obj=docs_obj
                )
            else:
                self.final_load(batched_docs=data, collection=c_etl, docs_obj=docs_obj)

            log.info(
                "{} : Total number of entities after the etl process to {} collection".format(
                    c_etl.num_entities, c_etl.name
                )
            )
            log.info("Swapping the alias from {} to {}".format(c_user.name, c_etl.name))
            utility.alter_alias(
                collection_name=c_etl.name, alias=self.config.collection_name
            )
        else:
            raise ValueError('pull_elastic parameter not valid value')
        # close any milvus connections that might be statically open
        for conn in connections.list_connections():
            mil_alias = conn[0]  # alias is first element in tuple
            log.info("disconnecting from " + mil_alias)
            connections.disconnect(mil_alias)

    def post_process(self) -> None:
        """Save results, etc."""

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['process_pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)