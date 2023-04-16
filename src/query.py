"""
This module provides a command line interface for querying a language model using a LLM and a Faiss index of embeddings
containing knowledge. It uses the Llama library to interact with the language model and the Faiss library to create
and search the index.

Note:
    If an index_path is provided, the function loads the Faiss index and searches it for the closest embeddings to
    the question embedding. It then prompts the user with a message that includes the context of the closest embedding
    and the original question.
"""


import logging
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from llama_cpp import Llama
from rich import print

from src.utils.Faiss import FaissIndex
from src.utils.logger import setup_logger

logger = setup_logger(__name__, logging.INFO)
llama: Optional[Llama] = None


def prompt_with_context() -> str:
    """
    Returns a prompt to request relevant text to answer a given question.

    Returns:
        str: A string prompt that includes the given context and question, and asks for relevant text.
    """
    return """
    Use the following portion of a long document to see if any of the text is relevant to answer the question.

    Context: {context}
    Question: {question}
    Provide all relevant text to the question verbatim. If nothing relevant return "I do not know" and stop answering.
    Answer:"""


def query(
    question: str = typer.Argument(..., help="Question to answer."),
    model_path: str = typer.Argument(..., help="Folder containing the model."),
    index_path: Optional[Path] = typer.Argument(
        None,
        help="Folder containing the vector store with the embeddings. If "
        "none provided, only LLM is used.",
    ),
):
    """
    Ask a question to a Language Model (LLM) using an index of embeddings containing the knowledge.

    If no `index_path` is specified, it will only use the LLM to answer the question. Otherwise, it will use the
    embeddings in the `index_path` to find relevant text before prompting for an answer.

    Args:
        question (str): The question to answer.
        model_path (str): The folder containing the LLM model.
        index_path (Optional[Path]): The folder containing the vector store with the embeddings. If none provided,
        only the LLM will be used.

    Returns:
        None: The response will be printed to the console.
    """
    global llama
    if not llama:
        if not model_path:
            raise Exception(
                "You need to specify the model path the first time you ask a question"
            )
        llama = Llama(model_path=model_path, verbose=False, embedding=True)
    if index_path:
        index = FaissIndex()
        index.load(index_path)
        embedded_question = np.array([llama.embed(question)])
        context = index.search(embedded_question)
        question = prompt_with_context().format(context=context[0], question=question)
    response = llama(prompt=question)["choices"][0]["text"]
    print(response.split('"""')[0])


if __name__ == "__main__":
    typer.run(query)
