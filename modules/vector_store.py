from langchain.document_loaders import  UnstructuredFileLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from typing import Any, Dict, List, Optional, Callable
from langchain.vectorstores import FAISS
import pickle
from langchain.docstore.base import Docstore
from pathlib import Path
from langchain.vectorstores.faiss import dependable_faiss_import


class ModifiedRecursiveCharacterTextSplitter(RecursiveCharacterTextSplitter):
    """Implementation of splitting text that looks at characters.

        Recursively tries to split by different characters to find one
        that works.
        """

    def __init__(self, separators: Optional[List[str]] = None, **kwargs: Any):
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        separators = separators or ["\n\n", "\n", ".", "?", "!", "... ", "。", "？", "！", "。。。。。。"]
        separators.append("")
        self.separators = separators

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        _good_splits = []
        # Get appropriate separator to use
        separator = self._separators[-1]
        for _s in self._separators:
            if _s == "":
                separator = _s
                break
            if _s in text:
                separator = _s
                break
        # Now that we have the separator, split the text
        if separator:
            splits = text.split(separator)

            # Now go merging things, recursively splitting longer texts.

            for s in splits:
                if self._length_function(s) < self._chunk_size:
                    _good_splits.append(s)
                else:
                    if _good_splits:
                        merged_text = self._merge_splits(_good_splits, separator)
                        final_chunks.extend(merged_text)
                        _good_splits = []
                    other_info = self.split_text(s)
                    final_chunks.extend(other_info)
        else:
            _good_splits.append(text)

        if _good_splits:
            merged_text = self._merge_splits(_good_splits, separator)
            final_chunks.extend(merged_text)
        return final_chunks


def text_splitter_generate(chunk_size: int = 5000, chunk_overlap: int = 200, chunk_type: str = 'sentence'):
    if chunk_type == 'paragraph':
        text_splitter = CharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=chunk_size,
                                              chunk_overlap=chunk_overlap)
    else:
        text_splitter = ModifiedRecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter


def loaders_to_texts(loaders, text_splitter=None):
    if text_splitter is None:
        text_splitter = ModifiedRecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    docs = text_splitter.split_documents(docs)
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    return texts, metadatas


def embedding_method_generate(model_name_or_path):
    """
    :param: model_name_or_path: If it is a filepath on disc, it loads the model from that path. If it is not a path, it first tries to download a pre-trained SentenceTransformer model. If that fails, tries to construct a model from Huggingface models repository with that name.
    :return:
    """
    return HuggingFaceEmbeddings(model_name=model_name_or_path)


class FaissVectorStore(FAISS):

    text_splitter = ModifiedRecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    VECTOR_SEARCH_TOP_K = 5

    def __init__(
        self,
        embedding_function: Callable,
        index: Any,
        docstore: Docstore,
        index_to_docstore_id: Dict[int, str],
    ):
        super().__init__(embedding_function, index, docstore, index_to_docstore_id)

    def save_local(self, folder_path: str) -> None:
        """Save FAISS index, docstore, and index_to_docstore_id to disk.

        Args:
            folder_path: folder path to save index, docstore,
                and index_to_docstore_id to.
        """
        path = Path(folder_path)
        path.mkdir(exist_ok=True, parents=True)

        # save index separately since it is not picklable
        faiss = dependable_faiss_import()
        faiss.write_index(self.index, str(path / "index.faiss"))

        # save docstore and index_to_docstore_id
        with open(path / "index.pkl", "wb") as f:
            pickle.dump((self.docstore, self.index_to_docstore_id, self.embedding_function), f)

    @classmethod
    def load_local(cls, folder_path: str) -> FAISS:
        """Load FAISS index, docstore, and index_to_docstore_id to disk.

        Args:
            folder_path: folder path to load index, docstore,
                and index_to_docstore_id from.
            embeddings: Embeddings to use when generating queries
        """
        path = Path(folder_path)
        # load index separately since it is not picklable
        faiss = dependable_faiss_import()
        index = faiss.read_index(str(path / "index.faiss"))

        # load docstore and index_to_docstore_id
        with open(path / "index.pkl", "rb") as f:
            docstore, index_to_docstore_id, embedding_function = pickle.load(f)
        return cls(embedding_function, index, docstore, index_to_docstore_id)


if __name__ == "__main__":
    loader = UnstructuredFileLoader('C:/Users/Administrator/Videos/Captures/2023.txt')
    texts = loaders_to_texts([loader, loader])
    embedding = embedding_method_generate('d:/model/retro-reader')
    vector_store = FaissVectorStore.from_texts(texts[0], embedding, texts[1])
    vector_store.save_local('d:/temp')
    vector_store = FaissVectorStore.load_local('d:/temp')
    print("OK!")
