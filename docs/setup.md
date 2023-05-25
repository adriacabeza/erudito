# How to Run

- The first step is to download the model weights in case you do not have them. There are a few torrents floating around as well as some huggingface repositories (e.g [https://huggingface.co/nyanko7/LLaMA-7B/](https://huggingface.co/nyanko7/LLaMA-7B/)). Once you have them, copy them into the models folder. Any of the quantized models that can run in llama.cpp should work: Llama, Alpaca, GPT4All, Vicuna...
- Install the necessary requirements available in the requirements.lock. The usage of a virtual environment is recommended.
```console
python -m venv env
source env/bin/activate
pip install -r requirements.lock
```
### **CLI**
There are two simple CLI applications:

- **ingest**: create an embedding vector store given a folder with a bunch of documents
```console
❯ python -m src.ingest --help

 Usage: python -m src.ingest [OPTIONS] DOCUMENTATION_PATH MODEL_PATH

 Ingest all the markdown files from documentation_path and create a vector store. It will
 create a new folder with the embedding index.

╭─ Arguments ───────────────────────────────────────────────────────────────────────────────╮
│ *    documentation_path      TEXT  Folder containing the documents. [default: None]       │
│                                    [required]                                             │
│ *    model_path              TEXT  Folder containing the model. [default: None]           │
│                                    [required]                                             │
╰───────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ─────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                               │
╰───────────────────────────────────────────────────────────────────────────────────────────╯
```

- **query**: make a question to the LLM given the embedding path
```console
❯ python -m src.query --help

 Usage: python -m src.query [OPTIONS] QUESTION MODEL_PATH [INDEX_PATH]

 Ask a question to a LLM using an index of embeddings containing the knowledge. If no
 index_path is specified it will only use the LLM to answer the question.

╭─ Arguments ───────────────────────────────────────────────────────────────────────────────╮
│ *    question        TEXT          Question to answer. [default: None] [required]         │
│ *    model_path      TEXT          Folder containing the model. [default: None]           │
│                                    [required]                                             │
│      index_path      [INDEX_PATH]  Folder containing the vector store with the            │
│                                    embeddings. If none provided, only LLM is used.        │
│                                    [default: None]                                        │
╰───────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ─────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                               │
╰───────────────────────────────────────────────────────────────────────────────────────────╯
```

### **API**

To run the uvicorn server:
```console
uvicorn src.api:app --reload --host 0.0.0.0 --port 8383
```
Then you can hit the two endpoints available:

**Ingest**:
```console
curl "http://0.0.0.0:8383/ingest?documentation_path=data&model_path=models/gpt4all/gpt4all-lora-quantized-new.bin"
```
**Query**:
```console
curl -X POST -H "Content-Type: application/json" --data '{
    "question":"who is the best?",
    "index_path":"index/datadog_documentation",
    "model_path":"models/gpt4all/gpt4all-lora-quantized-new.bin"}' http://0.0.0.0:8383/query```
```
Note that when hitting the query endpoint:

- The `index_path` is not mandatory. If not provided only the LLM will be used to answer the question.
- The `model_path` is only mandatory the first request. Once specified, the model will be already loaded in memory, and you will not need to provide a model path. If another `model_path` is provided a different model will be loaded.
