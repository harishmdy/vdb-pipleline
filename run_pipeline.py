import logging as log
import os
import resource
import sys

from ibm.unifiedsearchvectors.pipeline_builder.embeddings_pipeline import (
    EmbeddingsPipeline,
)
from ibm.unifiedsearchvectors.utils.functions import configure_logging, notify_slack

if __name__ == "__main__":
    try:
        configure_logging()
        config = {
            "doc_type": 'ibm_docs',
            "collection_name": 'ibm_docs_test_collection',
            "start_batch_num": 0,
            "pull_elastic": 'both',
            "is_html": 'True',
            "bulk_api": 0
        }

        if os.getenv('RUN_DOC_TYPE') is None:
            # log.error("RUN_DOC_TYPE missing from config")
            # sys.exit(1)
            raise Exception("RUN_DOC_TYPE missing from config")
        if os.getenv('COLLECTION_NAME') is None:
            # log.error("COLLECTION_NAME missing from config")
            # sys.exit(1)
            raise Exception("COLLECTION_NAME missing from config")

        limit_file = '/sys/fs/cgroup/memory/memory.limit_in_bytes'

        if os.path.isfile(limit_file):
            with open(limit_file) as limit:
                mem = int(limit.read())
                log.info(f'read current memoroty limit of {mem}')
                # take off .5 a MB to buffer other processes on pod
                buffer = 500000
                mem = mem - buffer
                log.info(resource.getrlimit(resource.RLIMIT_AS))
                log.info(f'Overriding RLIMIT_AS with {mem}')
                # resource.setrlimit(resource.RLIMIT_AS, (mem, mem))

        else:
            log.info(f'{limit_file} not found, leaving default')

        pipe = EmbeddingsPipeline(config)
        pipe.run()
    except Exception as err:
        log.exception(err)
        notify_slack(
            ':failed: \nThe Config: '
            + str(config)
            + " has failed with the following error.\n"
            + str(err)
        )
        sys.exit(1)
