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
    return """
    Use the following portion of a long document to see if any of the text is relevant to answer the question.
    {context}
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
    Ask a question to a LLM using an index of embeddings containing the knowledge. If no index_path is specified it
    will only use the LLM to answer the question.
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
