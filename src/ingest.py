import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import typer
from llama_cpp import Llama
from rich.progress import track

from src.utils.Faiss import FaissIndex
from src.utils.logger import setup_logger
from src.utils.reader import get_sources

# set up the logger
logger = setup_logger(__name__, logging.INFO)
llama: Optional[Llama] = None


def create_store(chunks: List[str], folder_name: str = "vector_store"):
    """
    Create the vector store
    """
    index = FaissIndex()
    embeddings = np.empty((len(chunks), 4096))
    index_path = Path("index") / folder_name / "index.faiss"

    # check if the index already exists
    if index_path.exists():
        logger.info("🦾 Updating existing index. This could take a while...")
        index.load(index_path)
    else:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("🚝 Create vector store. This could take a while...")

    count = 0
    for i in track(range(len(chunks)), description="Embedding 🦖"):
        try:
            embeddings[i] = llama.embed(chunks[i])
            count += 1
        except KeyboardInterrupt:
            logger.warning(f"❌ Stopped at {count} out of {len(chunks)}")
            index.add_vectors(embeddings, chunks)
            index.save(index_path)
            break
    # add the embeddings to the vector store
    index.add_vectors(embeddings, chunks)
    index.save(index_path)
    logger.info(f"🥒 Save FAISS vector store into a pickle in index/{folder_name}")


def ingest(
    documentation_path: str = typer.Argument(
        ..., help="Folder containing the documents."
    ),
    model_path: str = typer.Argument(..., help="Folder containing the model."),
):
    """
    Ingest all the text files from documentation_path and create a vector store. It will create a new folder with the embedding index.
    """
    global llama
    # initialize the Llama model
    if not llama:
        llama = Llama(model_path=model_path, embedding=True, verbose=False)
    logger.info(f"Look for docs in {documentation_path}")
    chunks = get_sources(documentation_path)
    if not chunks:
        raise Exception("No documents were found inside the data folder")
    create_store(chunks, Path(documentation_path).name)


if __name__ == "__main__":
    typer.run(ingest)
