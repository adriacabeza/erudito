import logging
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger(__name__, logging.INFO)


class FaissIndex:
    def __init__(
        self,
    ):
        self.index = None
        self.reverse_index = {}
        self.num_added = 0

    def load(self, index_file: Path) -> None:
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
        faiss.write_index(self.index, str(index_file))
        reverse_index_file = index_file.parent / "reverse_index.npy"
        np.save(str(reverse_index_file), self.reverse_index)

    def add_vectors(self, vectors: np.ndarray, contents: List[str]) -> None:
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
        assert self.index is not None, "Index has not been loaded yet"
        assert (
            query_embedding.shape[1] == self.index.d
        ), "Embedding dimensions do not match"
        distances, indices = self.index.search(query_embedding, 1)
        result_ids = [self.reverse_index[int(i)] for i in indices]
        return result_ids
