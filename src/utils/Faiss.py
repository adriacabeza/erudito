import logging
from pathlib import Path
from typing import List

import faiss
import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger(__name__, logging.INFO)


class FaissIndex:
    """
    A class for creating and querying a Faiss index.

    Attributes:
        index (faiss.Index): The Faiss index object used for similarity search.
        reverse_index (dict): A dictionary mapping document IDs to their corresponding
            index in the Faiss index. This allows for quick lookup of document vectors
            during query time.

    """

    def __init__(
        self,
    ):
        """
        Initializes an empty Faiss index and an empty reverse index.
        """
        self.index = None
        self.reverse_index = {}

    def load(self, index_file: Path) -> None:
        """
        Load the Faiss index and its corresponding reverse index from disk.

        Args:
            index_file (Path): The path to the Faiss index file.

        Raises:
            ValueError: If the index file or reverse index file is not found.

        """
        reverse_index_file = index_file.parent / "reverse_index.npy"
        if not index_file.exists():
            raise ValueError(f"No index found in {index_file}")
        elif not reverse_index_file.exists():
            raise ValueError(f"No reverse index found in {self.reverse_index}")
        else:
            self.index = faiss.read_index(str(index_file))
            self.reverse_index = np.load(
                str(reverse_index_file), allow_pickle=True
            ).item()

    def save(self, index_file: Path) -> None:
        """
        Save the Faiss index to disk.

        Args:
            index_file (Path): The path to the Faiss index file.
        """
        faiss.write_index(self.index, str(index_file))
        reverse_index_file = index_file.parent / "reverse_index.npy"
        np.save(str(reverse_index_file), self.reverse_index)

    def add_vectors(self, vectors: np.ndarray, contents: List[str]) -> None:
        """
        Add vectors and their corresponding contents to the Faiss index.

        Args:
            vectors (np.ndarray): The vectors to add to the index. Shape should be (num_vectors, vector_dim).
            contents (List[str]): The corresponding contents for each vector. Length should be num_vectors.

        Raises:
            ValueError: If the length of contents does not match the number of vectors.
        """
        if not self.index:
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(vectors.shape[1]))
            num_added = 0
        else:
            num_added = self.index.ntotal

        num_vectors = vectors.shape[0]
        ids = np.arange(num_added, num_added + num_vectors)
        self.index.add_with_ids(vectors, ids)
        for i in range(num_vectors):
            self.reverse_index[ids[i]] = contents[i]

    def search(self, query_embedding: np.ndarray) -> List[str]:
        """
        Search the Faiss index for the nearest neighbors of a given query embedding.

        Args:
            query_embedding (np.ndarray): An array of shape (1, D) containing the query embedding,
                where D is the dimensionality of the embeddings used to build the index.

        Returns:
            A list of strings, each of which corresponds to the content of the document that
            is closest to the query embedding in the embedding space.
        Raises:
            AssertionError: If the index has not been loaded yet.
            AssertionError: If the dimensionality of the query embedding does not match that of the index.
        """
        assert self.index is not None, "Index has not been loaded yet"
        assert (
            query_embedding.shape[1] == self.index.d
        ), "Embedding dimensions do not match"
        distances, indices = self.index.search(query_embedding, 1)
        result_ids = [self.reverse_index[int(i)] for i in indices]
        return result_ids
