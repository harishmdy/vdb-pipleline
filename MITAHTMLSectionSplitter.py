import copy
import importlib.resources
import logging as logger
import pathlib
from io import StringIO
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypedDict, cast

from langchain_core.documents import Document
from langchain_text_splitters import HTMLSectionSplitter

log = logger.getLogger("ibmsearch")


# class ElementType(TypedDict):
#     """Element type as typed dict."""

#     url: str
#     xpath: str
#     content: str
#     metadata: Dict[str, str] 


class MITAHTMLSectionSplitter(HTMLSectionSplitter):
    def __init__( 
        self,
        headers_to_split_on: List[Tuple[str, str]],
        xslt_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:

        super().__init__(
            headers_to_split_on=headers_to_split_on, xslt_path=xslt_path, **kwargs
        )
        try:
            from lxml import etree
        except ImportError as e:
            raise ImportError(
                "Unable to import lxml, please install with `pip install lxml`."
            ) from e
        # document transformation for "structure-aware" chunking is handled with xsl.
        # see comments in html_chunks_with_headers.xslt for more detailed information.
        # xslt_path = pathlib.Path(__file__).parent / "xsl/html_chunks_with_headers.xslt"

        # xslt_path_headers = importlib.resources.files("langchain_text_splitters").joinpath("xsl/html_chunks_with_headers.xslt")
        # xslt_tree_headers = etree.parse(xslt_path_headers)
        # self.transform_headers = etree.XSLT(xslt_tree_headers)

        # cpos commenting this out so we can serialize for multi-processing
        # self.xslt_tree_custom = etree.parse(self.xslt_path)
        # cpos commenting this out so we can serialize for multi-processing
        # self.transform_custom = etree.XSLT(xslt_tree_custom)

    """
    def convert_possible_tags_to_header(self, html_content: str) -> str:
        try:
            from lxml import etree
        except ImportError as e:
            raise ImportError(
                "Unable to import lxml, please install with `pip install lxml`."
            ) from e
        # use lxml library to parse html document and return xml ElementTree
        parser = etree.HTMLParser()
        tree = etree.parse(StringIO(html_content), parser)

        transform_custom = etree.XSLT(self.xslt_tree_custom)
        result = self.transform_custom(tree)
        return str(result)
    """

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            try:
                for chunk in self.split_text(text):
                    metadata = copy.deepcopy(_metadatas[i])

                    for key in chunk.metadata.keys():
                        if chunk.metadata[key] == "#TITLE#":
                            chunk.metadata[key] = metadata["Title"]
                    metadata = {**metadata, **chunk.metadata}
                    new_doc = Document(
                        page_content=chunk.page_content, metadata=metadata
                    )
                    documents.append(new_doc)
            except ValueError:
                log.debug("Skipping document in HTML splitter")
                # print("skipping document in HTML splitter")
            except Exception as e:
                log.error(f"{e} on document {_metadatas[i]['url']}")

        return documents
