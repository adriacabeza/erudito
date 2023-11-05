import logging
from pathlib import Path
from typing import List

import markdown
import PyPDF2
from bs4 import BeautifulSoup

from src.utils.logger import setup_logger

logger = setup_logger(__name__, logging.INFO)


def split_text(text: str, separator: str = " ", chunk_size: int = 512) -> List[str]:
    """
    Splits a text string into chunks of at most chunk_size characters,
    using the specified separator character (default is space).
    Returns a list of strings.

    Args:
        text: str, the text string to be split.
        separator: str, the character to use as a separator for splitting. Defaults to ' ' (space).
        chunk_size: int, the maximum length of each chunk. Defaults to 512.

    Returns:
        List[str], a list of string chunks of the original text.

    Raises:
        Exception: if the text cannot be split into chunks of chunk_size characters using the specified separator.
    """
    chunks = []
    start = 0
    end = 0
    while end < len(text):
        end = min(start + chunk_size, len(text))
        if end == len(text):
            chunks.append(text[start:end])
            break
        if text[end] == separator:
            chunks.append(text[start:end])
            start = end + 1
        elif text[start:end].count(separator) == 0:
            chunks.append(text[start:end])
            start = end
        else:
            last_separator = text[start:end].rfind(separator)
            if not last_separator:
                raise Exception(
                    f"This text cannot be split into chunks of {chunk_size} characters "
                    f"using the selected separator."
                )
            chunks.append(text[start : start + last_separator])
            start += last_separator + 1
    return chunks


def pdf_to_text(input_file: Path) -> str:
    """
    Convert a PDF file to plain text.

    Args:
        input_file: Path, the path to the input PDF file.

    Returns:
        str, the plain text content of the PDF file.
    """
    with open(input_file, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text


def markdown_to_text(text: str) -> str:
    """
    Convert a Markdown string to plain text.

    Args:
        text: str, the Markdown string to be converted.

    Returns:
        str, the plain text content of the Markdown string.
    """
    html = markdown.markdown(text)
    soup = BeautifulSoup(html, features="html.parser")
    return soup.get_text()


def get_text(input_file: Path) -> str:
    """
    Determines the file extension and converts the file to plain text.

    Args:
        input_file (Path): The path to the file to convert.

    Returns:
        str: The plain text content of the file.

    Raises:
        Exception: If the file extension is not supported or the file cannot be read.
    """
    ext = input_file.suffix.lower()
    if ext == ".pdf":
        return pdf_to_text(input_file)
    elif ext == ".md":
        with open(input_file) as f:
            return markdown_to_text(f.read())
    else:
        try:
            with open(input_file) as f:
                return f.read()
        except:
            logger.info(f"Skipping {input_file} because it is not a text file")
            return ""


def get_sources(documentation_path: str) -> List[str]:
    """
    Get the plain text of all the documents in the given folder and split them into smaller chunks.

    Args:
        documentation_path (str): Path to the directory containing the documents.

    Returns:
        List of strings: A list of all the documents, split into smaller chunks.
    """
    ps = list(Path(documentation_path).glob("**/*.*"))
    logger.info(f"📖 {len(ps)} documents were found")

    data = []
    sources = []
    for p in ps:
        text = get_text(p)
        if text:
            data.append(text)
            sources.append(p)

    # Split the documents, as needed, into smaller chunks.
    # We do this due to the context limits of the LLMs.
    docs = []
    for i, d in enumerate(data):
        splits = split_text(d)
        docs.extend(splits)

    return docs
