import logging
from pathlib import Path

import markdown
import nltk
import PyPDF2
from bs4 import BeautifulSoup

from src.utils.logger import setup_logger

nltk.download("punkt")
from typing import List

logger = setup_logger(__name__, logging.INFO)


def split_text(text: str) -> List[str]:
    """
    Split a text into chunks of 512 characters or fewer.
    """
    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)

    # Initialize an empty list to store the chunks
    chunks = []

    # Loop through each sentence
    for sentence in sentences:
        # Split the sentence into chunks of 512 characters or fewer
        sentence_chunks = [sentence[i : i + 512] for i in range(0, len(sentence), 512)]

        # Merge contiguous chunks that are less than 512 characters
        merged_chunks = []
        current_chunk = ""
        for chunk in sentence_chunks:
            if len(current_chunk + chunk) <= 512:
                current_chunk += chunk
            else:
                merged_chunks.append(current_chunk)
                current_chunk = chunk
        merged_chunks.append(current_chunk)

        # Add each merged chunk to the list of chunks
        chunks.extend(merged_chunks)
    return chunks


def pdf_to_text(input_file: Path) -> str:
    """
    Convert a PDF file to plain text.
    """
    with open(input_file, "rb") as f:
        reader = PyPDF2.PdfFileReader(f)
        text = ""
        for i in range(reader.getNumPages()):
            page = reader.getPage(i)
            text += page.extractText()
    return text


def markdown_to_text(text: str) -> str:
    """
    Convert a Markdown file to plain text.
    """
    html = markdown.markdown(text)
    soup = BeautifulSoup(html, features="html.parser")
    return soup.get_text()


def get_text(input_file: Path) -> str:
    """
    Determine the file extension and convert to plain text accordingly.
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
