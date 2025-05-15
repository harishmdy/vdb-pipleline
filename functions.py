import ast
import importlib.resources
import io
import json
import logging as logger
import os
import sys
import time
from datetime import datetime, timezone
from typing import Callable, Optional

import boto3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from pymilvus import (
    BulkInsertState,
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from scrapy.utils.project import get_project_settings

from ibm.unifiedsearchvectors.utils.config import ObjectStorageConfig
from ibm.unifiedsearchvectors.utils.connect import CosManager
from ibm.unifiedsearchvectors.utils.elastic import SafeElasticsearch

MITA_ACCESS_KEY = os.getenv("MITA_AWS_ACCESS_KEY_ID", "")
MITA_SECRET_ACCESS_KEY = os.getenv("MITA_AWS_SECRET_ACCESS_KEY", "")
MITA_BUCKET = os.getenv("MITA_AWS_BUCKET", "")
MITA_ENDPOINT = os.getenv("MITA_AWS_ENDPOINT")
SLACK_CHNL_WEBHOOK_URL = os.getenv("SLACK_ALERT_WEBHOOK_URL", "")
# MITA_SLACK_BOT_TOKEN = os.getenv("MITA_SLACK_BOT_TOKEN", "")
MILVUS_ACCESS_KEY = os.getenv("MILVUS_COS_ACCESS_KEY_ID", "")
MILVUS_SECRET_ACCESS_KEY = os.getenv("MILVUS_COS_SECRET_ACCESS_KEY", "")
MILVUS_COS_BUCKET = os.getenv("MILVUS_COS_BUCKET", "")
MILVUS_COS_ENDPOINT = os.getenv("MILVUS_COS_ENDPOINT", "")
N_SCROLL = int(os.getenv("N_SCROLL", "30"))

COS_ENV = os.getenv("COS_ENV", "test")
COS_POST_FIX = os.getenv("COS_POST_FIX", "")

log = logger.getLogger("ibmsearch")


def build_cos_parquet_path(base_file: str):
    return f"vec_db/pipeline/{COS_ENV}/{base_file}{COS_POST_FIX}.parquet"


def build_cos_csv_path(base_file: str):
    return f"vec_db/pipeline/{COS_ENV}/{base_file}{COS_POST_FIX}.csv"


def connect_cos():
    obj_config = CosManager(
        ObjectStorageConfig(
            access_key=MITA_ACCESS_KEY,
            secret_key=MITA_SECRET_ACCESS_KEY,
            bucket=MITA_BUCKET,
            endpoint_url=MITA_ENDPOINT,
        )
    )
    return obj_config


def connect_milvus_cos():
    obj_config = CosManager(
        ObjectStorageConfig(
            access_key=MILVUS_ACCESS_KEY,
            secret_key=MILVUS_SECRET_ACCESS_KEY,
            bucket=MILVUS_COS_BUCKET,
            endpoint_url=MILVUS_COS_ENDPOINT,
        )
    )
    return obj_config


def connect_milvus():
    connection_params = {
        "db_name": os.getenv("MLVS_DB_NM", ""),
        "alias": os.getenv("MLVS_DB_ALIAS", "default"),
        "secure": (os.getenv("MLVS_SECURE", "True") == "True"),
        "user": os.getenv("MLVS_UID", ""),
        "password": os.getenv("MLVS_PWD", ""),
        "host": os.getenv("MLVS_HOST", ""),
        "port": os.getenv("MLVS_PORT", ""),
        "server_name": os.getenv("MLVS_SERVER_NAME", ""),
        "server_pem_path": os.getenv("MLVS_CERT_PATH", ""),
    }

    connection_params = {k: v for k, v in connection_params.items() if v}
    connections.connect(**connection_params)


def convert_datetime(text):
    last_e: Exception
    for date_format in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            date_obj = datetime.strptime(text, date_format)
            return int(date_obj.replace(tzinfo=timezone.utc).timestamp() * 1000)
        except (ValueError, TypeError) as v:
            last_e = v
            if (text is None) or (text == ""):
                return int(0)
            # log.exception(v)
            pass

    raise ValueError("no valid date format found", text, last_e)


def truncate_list(x, integer):
    try:
        if isinstance(x, list):
            return x[:integer]
        elif isinstance(x, np.ndarray):
            return x.tolist()[:integer]
        else:
            return []
    except Exception as e:
        log.warning(str(e))
        return []


def truncate_list_elements(x, integer):
    try:
        return [s[:integer] for s in x]
    except Exception as e:
        # log.exception(e)
        return []


def collapse_dictionary(x, column_to_log="column"):
    try:
        if x is not None:
            return dict(ast.literal_eval(x))
    except ValueError as e:
        log.warning("Value error collapsing " + column_to_log + " for " + str(x))
    except SyntaxError as s:
        log.warning("Syntax error collapsing " + column_to_log + " for " + str(x))
    except TypeError as t:
        log.warning("Type error collapsing " + column_to_log + " for " + str(x))
    return {}


def configure_logging():
    logger.basicConfig(
        level=logger.INFO,
        #    format="%(asctime)s [%(levelname)s] %(message)s",
        format="%(asctime)s %(levelname)s %(process)d - %(message)s",
        handlers=[
            # log.FileHandler(f"pipe_{datetime.today().strftime('%Y-%m-%d')}.log"),
            logger.StreamHandler(stream=sys.stdout),
        ],
    )
    logger.getLogger("ibmsearch").setLevel(logger.INFO)


def cast_to_list(x):
    #    if isinstance(x, list) or isinstance(x, np.ndarray):
    if isinstance(x, list):
        return x
    else:
        return [x]

def query_elastic(
    query_dict: dict,
    cleaner_func: Optional[Callable],
    adopter_specific_keys=(
        "dwcontenttype",
        "ibmdocstype",
        "ibmdocsproduct",
        "tsdoctypedrill",
        "ibm_tssoftware_version_original",
        "ibmdocskey",
        "tscategory",
    ),
    partition_callback: Optional[Callable[[int, pa.Table], None]] = None,
    partition_batch_size: int = N_SCROLL
) -> Optional[pa.Table]:

    es = SafeElasticsearch(
        host=os.getenv("ELASTIC_HOST", ""),
        index=os.getenv("ELASTIC_INDEX", ""),
        api_key=os.getenv("ELASTIC_API_KEY", ""),
        cert_path=os.getenv("ELASTIC_CERT_PATH", ""),
        max_calls=2,
    )

    response = es.search(size=2, body=query_dict)
    columns = query_dict["_source"]
    scroll_id = response["_scroll_id"]
    results_tb = None
    scroll_count = 0
    partition_index = 0
    start = time.time()

    count = 0 
    while len(response['hits']['hits']) and (count < 2):
    # while len(response["hits"]["hits"]):
        count += 1

        results = []
        for hit in response["hits"]["hits"]:
            if "adopter_specific" in hit["_source"]:
                hit["_source"]["adopter_specific"] = {
                    x: hit["_source"]["adopter_specific"][x]
                    for x in adopter_specific_keys
                    if x in hit["_source"]["adopter_specific"]
                }
            results.append(hit["_source"])

        tmp_df = pd.DataFrame.from_records(results, columns=columns)
        if cleaner_func:
            tmp_df = cleaner_func(tmp_df)

        tmp_result_tb = pa.Table.from_pandas(tmp_df)
        scroll_count += 1

        if partition_callback:
            if results_tb is None:
                results_tb = tmp_result_tb
            else:
                results_tb = pa.concat_tables([results_tb, tmp_result_tb], promote_options="permissive")

            if scroll_count % partition_batch_size == 0:
                partition_callback(partition_index, results_tb)
                partition_index += 1
                results_tb = None  # reset buffer
        else:
            if results_tb is None:
                results_tb = tmp_result_tb
            else:
                results_tb = pa.concat_tables([results_tb, tmp_result_tb], promote_options="permissive")

        response = es.scroll(scroll_id=scroll_id, scroll="10m")

    # Final flush
    if partition_callback and results_tb is not None:
        partition_callback(partition_index, results_tb)

    es.clear_scroll(scroll_id)
    es.close()
    log.info(f"Elastic pull time: {time.time() - start}")
    return None if partition_callback else results_tb



def check_alias(collection: str):
    # check to see which is being pointed to
    if collection in utility.list_aliases(collection + "_green"):
        c_user = Collection(collection + "_green")
        c_etl = Collection(collection + "_blue")
        log.info(
            "{} is the active collection with {} entities".format(
                c_user.name, c_user.num_entities
            )
        )
        log.info(
            "{} is the collection with {} entities to process etl ".format(
                c_etl.name, c_user.num_entities
            )
        )
    else:
        c_user = Collection(collection + "_blue")
        c_etl = Collection(collection + "_green")
        log.info(
            "{} is the active collection with {} entities".format(
                c_user.name, c_user.num_entities
            )
        )
        log.info(
            "{} is the collection with {} entities to process etl ".format(
                c_etl.name, c_etl.num_entities
            )
        )
    return c_user, c_etl


def create_alias(docs_obj, collection_name: str):
    # Create collections/alias
    names = [collection_name + "_green", collection_name + "_blue"]
    utility.list_collections()
    if (names[0] not in utility.list_collections()) and (
        names[1] not in utility.list_collections()
    ):
        # if (~utility.has_collection(names[0])) and (~utility.has_collection(names[1])):
        for name in names:
            log.info("create alias, reset to green for " + name)
            docs_obj.create_collection(name, 384)
            utility.drop_alias(alias=collection_name)
            utility.create_alias(
                collection_name=collection_name + "_green", alias=collection_name
            )


def get_schema_comments(docs_type=""):
    schema_comments_config_dict = {}
    # return_comments_dict = {}
    with importlib.resources.path(
        "ibm.unifiedsearchvectors.resources", "schema_comments.json"
    ) as f_name:
        with f_name.open("rt") as f:
            schema_comments_config_dict = json.load(f)

    field_info_common = schema_comments_config_dict.get("common").get("fields")
    collection_desc = ""
    # if there are specific fields, merge them into the common ones.
    if (docs_type != "") & (docs_type in schema_comments_config_dict):
        field_info_common.update(
            schema_comments_config_dict.get(docs_type).get("fields")
        )
        collection_desc = schema_comments_config_dict.get(docs_type).get("collection")

    return {"collection": collection_desc, "fields": field_info_common}


def retry_request(url, payload, headers, total=3):
    """
    # retry_request
    retry_request is a function that attempts to send a POST request to the specified URL with the given payload and headers. It will retry the request up to total times if there is a connection error or if the response status code is not 200. If the response contains an "errors" key, it will also retry. The function returns the response if successful, otherwise it returns None.

    ## Parameters
    url: The URL to send the POST request to.
    payload: The data to send in the request body.
    headers: The headers to send with the request.
    total: (optional) The maximum number of times to retry the request. Default is 3.
    ## Attributes
    requests: The requests library is used to make the HTTP requests.
    ## Return Type
    The function returns a requests.Response object if the request is successful, otherwise it returns None.

    ## Exceptions and Edge Cases
    If the request fails more than total times, the function will return None.
    If the response status code is not 200, the function will retry the request.
    If the response contains an "errors" key, the function will retry the request.
    """
    for _ in range(total):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=300)
            if response.status_code != 200:
                log.warning(
                    f"Failed with status code {response.status_code}...Retrying url '{url}' with payload '{payload}'"
                )
                continue
            if "errors" in response.json().keys():
                log.warning(
                    f"Errors detected in response...Retrying url '{url}' with payload '{payload}'"
                )
                continue
            return response
        except requests.exceptions.ConnectionError:
            pass
        return None


def fetch_graphql_token():
    endpoint_url = os.getenv(
        "SSO_ENDPOINT_URL",
        "https://sso.redhat.com/auth/realms/redhat-external/protocol/openid-connect/token",
    )
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = (
        "grant_type=client_credentials&scope=openid api.graphql&client_id="
        + os.getenv("SSO_CLIENT", "")
        + "&client_secret="
        + os.getenv("SSO_SECRET", "")
    )

    response = requests.post(endpoint_url, headers=headers, data=data, timeout=300)

    if response.status_code == 200:
        response_json = response.json()
        return response_json["access_token"]
    else:
        log.exception(
            f"POST request to fetch token failed with status code, {response.status_code}"
        )
        return response


def execute_queries(queries, variables, graphql_token):
    """
    # execute_queries
    execute_queries is a function that takes in a list of GraphQL queries and their corresponding variables as input, and executes them using the provided GraphQL token. It returns a list of JSON responses from the GraphQL API.

    ## Parameters
    queries: A list of GraphQL queries to be executed.
    variables: A list of variables corresponding to each query in queries.
    graphql_token: The authentication token used to authorize the GraphQL requests.
    ## Return Value
    A list of JSON responses from the GraphQL API.

    ## Exceptions and Edge Cases
    The function may raise the following exceptions or edge cases:

    If the length of queries and variables are not equal, the function will raise a ValueError.
    If the GraphQL request fails, the function will log a warning message and continue processing the remaining requests.
    """

    url = os.getenv("RH_GRAPHQL_ENDPOINT", "https://graphql.redhat.com/")

    headers = {
        "apollographql-client-name": os.getenv("RH_GRAPHQL_CLIENT_NAME", "ibm"),
        "apollographql-client-version": os.getenv(
            "RH_GRAPHQL_CLIENT_VERSION", "latest"
        ),
        "content-type": os.getenv("RH_GRAPHQL_CONTENT_TYPE", "application/json"),
        "Authorization": f"Bearer {graphql_token}",
    }

    responses = []

    for idx in range(len(queries)):
        payload = {"query": queries[idx], "variables": variables[idx]}
        response = retry_request(url, payload=payload, headers=headers)
        if response is None:
            log.warning(
                f"Request with query '{queries[idx]}' and variable '{variables[idx]}' returned None"
            )
            continue

        if response.status_code == 200:
            response_json = response.json()

            if len(response_json) == 0:
                log.warning(
                    f"Request with query '{queries[idx]}' and variable '{variables[idx]}' is empty"
                )
                continue
            responses.append(response_json)

            if "errors" in response_json.keys():
                log.warning(
                    f"Error with query '{queries[idx]}' and variable '{variables[idx]}', message: {response_json['message']}"
                )
        else:
            log.warning(
                f"POST request with query '{queries[idx]}' and variable '{variables[idx]}' failed with status code: ",
                response.status_code,
            )
            continue
    return responses


def parse_slugs(json):
    """
    # parse_slugs
    This function parses the slugs from the JSON response returned by the GraphQL API call to get the list of products and their versions. It returns two dictionaries: one mapping product URL slugs to product names, and another containing a list of dictionaries with product URL slugs and version URL slugs.

    ## Parameters
    json: This is the JSON response returned by the GraphQL API call.
    ## Return Values
    product_mapping: A dictionary that maps product URL slugs to product names.
    filters: A list of dictionaries containing product URL slugs and version URL slugs.
    ## Exceptions and Edge Cases
    If the JSON response does not have the expected structure, a KeyError exception will be raised. The error message will be "invalid json format".
    If any of the required keys are missing in the JSON response, the corresponding values will be set to None.
    If any of the required sub-keys are missing in the JSON response, the corresponding values will be set to None.
    If the documentation key is present but its value is None, the corresponding values will be set to None.
    If the version key is present but its value is None, the corresponding values will be set to None
    """

    filters = []
    product_mapping = {}

    try:
        products_search_edges = json["data"]["products_search"]["edges"]
    except KeyError as e:
        log.exception("invalid json format")
        raise KeyError("invalid json format") from e

    for node_dict in products_search_edges:
        product_dict = {}
        node = node_dict["node"]

        if "name" not in node.keys():
            continue

        if "documentation" not in node.keys():
            continue
        else:
            product_name = node["name"]
            documentation = node["documentation"]

            if documentation is None:
                continue
            else:
                if ("slug" not in documentation.keys()) | (
                    "version" not in documentation.keys()
                ):
                    continue
                else:
                    product_url_slug = documentation["slug"]

                    if "slug" not in documentation["version"].keys():
                        continue
                    else:
                        product_version_slug = documentation["version"]["slug"]

                        product_dict["productUrlSlug"] = product_url_slug
                        product_dict["productVersionUrlSlug"] = product_version_slug

                        filters.append(product_dict)

                        product_mapping[product_url_slug] = product_name
    return product_mapping, filters


def parse_documentation(json, product, version) -> pd.DataFrame:
    """
    # parse_documentation
    This function parses the JSON response from the GraphQL API query for documentation titles and returns a Pandas DataFrame containing the documentation title, product, version, and other relevant metadata.

    ## Parameters
    json: The JSON response from the GraphQL API query.
    product: The name of the product for which documentation is being parsed.
    version: The version of the product for which documentation is being parsed.
    ## Returns
    A Pandas DataFrame containing the documentation title, product, version, and other relevant metadata.

    ## Exceptions and Edge Cases
    If the JSON response does not contain any documentation titles, an empty Pandas DataFrame will be returned.
    """
    parsed_nodes = []
    edges = json["data"]["documentation_titles"]["edges"]

    for node in edges:
        parsed_node = parse_node(node, product, version)
        parsed_nodes.append(parsed_node)

    if parsed_nodes:
        parsed_nodes_df = pd.concat(parsed_nodes).reset_index(drop=True)
    else:
        parsed_nodes_df = pd.DataFrame()
    return parsed_nodes_df


def parse_node(json, product, version):
    """
    # parse_node
    Parses a JSON node into a Pandas DataFrame.

    ## Parameters
    json: The JSON object to be parsed.
    product: The name of the product being parsed.
    version: The version of the product being parsed.
    ## Returns
    A Pandas DataFrame containing the parsed data.

    ## Exceptions
    KeyError: Raised if the JSON node is not in the expected format or does not contain the required keys.
    """

    try:
        node = json["node"]
    except KeyError as e:
        log.exception("Node is not the expected format")
        raise KeyError("Node is not the expected format") from e

    try:
        title_name = node["name"]
        title_url = node["url"]
        description = node["description"]
        multipages = node["multiPages"]
    except KeyError as e:
        log.exception(
            "Node must contain 'name', 'description', 'url', and 'multiPages' as keys"
        )
        raise KeyError(
            "Node must contain 'name', 'description', 'url', and 'multiPages' as keys"
        ) from e

    df = pd.DataFrame(multipages).rename(
        {"name": "page_name", "contentUrl": "page_raw_content_url"}, axis=1
    )
    metadata_cols = [description, title_name, title_url, version, product]
    metadata_colnames = ["description", "title", "title_url", "version", "product"]

    for col_name, col in zip(metadata_colnames, metadata_cols):
        df.insert(0, col_name, col)

    return df


def repeat_query(query_template, n):
    return np.repeat(query_template, n)


def make_variables(filters, first=500):
    variables = []
    for product in filters:
        variable_dict = {}
        variable_dict["first"] = first
        variable_dict["filter"] = product

        variables.append(variable_dict)

    return variables


def keyword_filter(products_filters, keyword):
    """
    # Keyword Filter
    ## Purpose
    The purpose of this function is to filter a list of dictionaries based on a keyword. The function takes in two parameters: products_filters which is a list of dictionaries and keyword which is a string. The function returns a new list of dictionaries that contain the keyword in the "productUrlSlug" key.

    ## Attributes
    products_filters: A list of dictionaries containing product information.
    keyword: A string representing the keyword to search for.
    ## Return Type
    A list of dictionaries containing product information that match the keyword.

    ## Exceptions and Edge Cases
    If the products_filters parameter is not a list, the function will raise a TypeError.
    If the keyword parameter is not a string, the function will raise a TypeError.
    If the products_filters parameter is an empty list, the function will return an empty list.
    If the keyword parameter is an empty string, the function will return the original products_filters list.
    """
    keyword = keyword.lower()
    filtered_products_filters = []
    for product_dict in products_filters:
        if keyword in product_dict["productUrlSlug"].lower():
            filtered_products_filters.append(product_dict)
    return filtered_products_filters


def find_differing_columns(df):
    """
    # find_differing_columns
    ## Description
    This function takes in a pandas dataframe and returns a list of columns that have values that differ between rows.

    ## Parameters
    df: A pandas dataframe.
    ## Returns
    A list of strings representing the names of the columns that have values that differ between rows.

    ## Exceptions and Edge Cases
    If the input is not a pandas dataframe, it will raise a TypeError.
    If the dataframe is empty, it will return an empty list.
    """
    differing_columns = []
    for i in range(len(df) - 1):
        for j in range(i + 1, len(df)):
            differing_cols = []
            for column in df.columns:
                if df.at[i, column] != df.at[j, column]:
                    differing_cols.append(column)
                differing_columns.extend(differing_cols)
    return list(set(differing_columns))


def dedup_redhat(metadata_df):
    """
    # dedup_redhat
    Deduplicate a dataframe of Red Hat content.

    ## Parameters
    metadata_df: A pandas dataframe containing the metadata for Red Hat content.
    ## Returns
    df: A pandas dataframe containing the deduplicated metadata.
    ## Functionality
    This function takes a pandas dataframe containing metadata for Red Hat content and deduplicates it based on the following criteria:

    1. If two or more rows have the same URL, they are considered duplicates and we will try and deduplicate so only one row is kept.
    2. If two or more rows have different products, we are unable to deduplicate and both instances are kept.
    3. If two or more rows have different metadata other than product, we will deduplicate based on the length of the title_url. We will keep the
    row with the shortest title_url
    4. After deduplication, we log a summary of the deduplication cases and the function returns a new dataframe with the deduplicated data.
    """
    log.info("Beginning deduping")
    metadata_df_copy = metadata_df.copy()
    metadata_df_copy = metadata_df_copy.drop_duplicates().reset_index(drop=True)
    metadata_df_copy["title_url_len"] = metadata_df_copy["title_url"].apply(len)

    duplicated_df = metadata_df_copy[
        metadata_df_copy.duplicated(subset=["url"], keep=False)
    ]
    duplicated_urls = duplicated_df["url"].unique()

    nonduplicated_df = metadata_df_copy.loc[
        ~metadata_df_copy.index.isin(list(duplicated_df.index))
    ]

    diff_product_count = 0
    dedup_count = 0
    bad_dedup_count = 0

    deduped_df = []
    for url in duplicated_urls:
        temp = duplicated_df[duplicated_df["url"] == url].reset_index(drop=True)
        diff_cols = find_differing_columns(temp)

        if "product" in diff_cols:
            deduped_df.append(temp)
            diff_product_count += 1
        else:
            if "title_url_len" in diff_cols:
                deduped_df.append(temp.loc[[temp["title_url_len"].idxmin()]])
                dedup_count += 1
            else:
                deduped_df.append(temp)
                bad_dedup_count += 1

    log.info(
        f"Successfully deduped: {dedup_count} URLs\n"
        f"Product is different, unable to dedup: {diff_product_count} URLs\n"
        f"Could not dedup using existing rules: {bad_dedup_count} URLs"
    )

    if len(deduped_df) == 0:
        deduped_df = pd.DataFrame()
    else:
        deduped_df = pd.concat(deduped_df).reset_index(drop=True)

    df = pd.concat([nonduplicated_df, deduped_df])
    df = df.reset_index(drop=True)
    df = df.drop("title_url_len", axis=1)

    return df


def get_scrapy_settings():
    settings = {"REQUEST_FINGERPRINTER_IMPLEMENTATION": "2.7"}
    return settings


def disable_propagate_logging():
    botocore_logs = [
        "botocore.hooks",
        "botocore.endpoint",
        "botocore.endpoint",
        "botocore.auth",
        "botocore.parser",
        "botocore.parsers",
        "botocore.awsrequest",
        "botocore.httpsession",
        "botocore.regions",
        "botocore.handlers",
        "botocore.retries.standard",
    ]
    for log_name in [
        "chardet.charsetprober",
        "scrapy",
        "urllib3.connectionpool",
    ] + botocore_logs:
        logger.getLogger(log_name).propagate = False


def remove_global_noise_df(
    bodies: "pd.Series[str]", global_noise: list[list[str]]
) -> "pd.Series[str]":
    cleaned_bodies = bodies

    for pat, repl in global_noise:
        cleaned_bodies = cleaned_bodies.str.replace(pat, repl, regex=True)

    return cleaned_bodies


def remove_local_noise_df(
    bodies: "pd.Series[str]",
    urls: "pd.Series[str]",
    local_noise: dict[str, list[str]],
) -> "pd.Series[str]":
    cleaned_bodies = bodies

    for url_pat, noise_list in local_noise.items():
        matched: "pd.Series[bool]" = urls.str.match(url_pat, na=False)

        for noise in noise_list:
            cleaned_bodies.loc[matched] = cleaned_bodies.loc[matched].str.replace(
                noise, "", regex=False
            )

    return cleaned_bodies


# Function to send slack notifictions
def notify_slack(message):
    json_data = {
        "text": message,
        "emoji": "",
        # "Authorization": "Bearer " + MITA_SLACK_BOT_TOKEN,
    }
    response = requests.post(SLACK_CHNL_WEBHOOK_URL, json=json_data)
    return response


def format_json(embeddings, metadata, page_content):
    assert len(embeddings) == len(metadata) and len(embeddings) == len(
        page_content
    ), "embeddings, metadata, and page_content must be the same length"
    json_dict = {"rows": []}
    for embedding, metadatum, content in zip(embeddings, metadata, page_content):
        row = metadatum
        row['content'] = content
        row['doc_vector'] = list(embedding)

        for key, obj in row.items():

            if isinstance(obj, np.integer):
                row[key] = int(obj)
            elif isinstance(obj, np.floating):
                row[key] = float(obj)
            elif isinstance(obj, np.ndarray):
                row[key] = obj.tolist()

        json_dict["rows"].append(row)

    return json_dict


def wait_tasks(
    task_id_dict, state_code, log_completion=False, timeout=None, retry_limit=3
):
    """
    Waits for tasks to reach a certain state, with optional logging and timeout.

    Parameters:
        - task_id_dict: A dictionary mapping task IDs to their corresponding data.
        - state_code: The state code to wait for.
        - log_completion: If True, logs information about completed tasks.
        - timeout: Maximum time to wait in seconds (optional). If None, waits indefinitely.
        - retry_limit: Maximum number of retries for failed tasks.

    Returns:
        A list of task states.
    """

    task_ids = task_id_dict.keys()
    wait_ids = list(task_ids)
    states = []
    complete_count = 0
    retries = {id: 0 for id in wait_ids}  # Tracks retry attempts for failed tasks
    start_time = time.time()  # Track the start time for timeout purposes

    while wait_ids:
        time.sleep(2)  # Wait once before gathering states

        # Check for timeout if specified
        if timeout and (time.time() - start_time) > timeout:
            log.warning("Timeout reached. Stopping the task state wait.")
            break

        temp_ids = []
        for id in wait_ids:
            state = utility.get_bulk_insert_state(task_id=id)

            # Handle failed tasks with retry logic
            if (
                state.state == BulkInsertState.ImportFailed
                or state.state == BulkInsertState.ImportFailedAndCleaned
            ):
                retries[id] += 1
                if retries[id] > retry_limit:
                    log.error(
                        f"Batch number {task_id_dict[state.task_id][0]}, task id: {id} failed after {retry_limit} retries. Reason: {state.failed_reason}"
                    )
                    raise Exception(
                        f"Batch number {task_id_dict[state.task_id][0]}, task id: {id} failed after {retry_limit} retries."
                    )

                else:
                    log.warning(
                        f"Batch number {task_id_dict[state.task_id][0]}, task id: {id} failed. Retrying {retries[id]} of {retry_limit}... Reason: {state.failed_reason}"
                    )
                    temp_ids.append(id)  # Retry this task
                continue

            # If the task reaches the desired state, append it to states
            if state.state >= state_code:
                states.append(state)
                if log_completion and state.state == BulkInsertState.ImportCompleted:
                    complete_count += 1
                    log.info(
                        f"Batch number {task_id_dict[state.task_id][0]}, task id: {state.task_id} successfully loaded with {task_id_dict[state.task_id][1]} embeddings!"
                    )
                continue

            temp_ids.append(id)

        wait_ids = temp_ids  # Update the list of tasks to check in the next iteration

    if log_completion:
        log.info(
            f"{complete_count} of {len(task_id_dict)} tasks have successfully generated segments, able to be compacted and indexed as normal."
        )

    return states