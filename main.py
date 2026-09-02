from pathlib import Path

import torch
from gensim.models import Doc2Vec

from classify import classify as data_classify
from cleaner import cleaner as data_cleaner
from crawler import crawler as data_crawler
from embedding import embedding as data_embedding
from tokenizer import tokenizer as data_tokenizer

BOARD_NAMES = [
    "Baseball",
    "Boy-Girl",
    "C_Chat",
    "HatePolitics",
    "Lifeismoney",
    "Military",
    "PC_Shopping",
    "Stock",
    "Tech_Job",
]

REQUIRED_TITLE_COUNT = 10000
CLEANED_DATA_FILE_PATH = "cleaned_data.csv"
TOKENIZED_DATA_FILE_PATH = "tokenized_data.csv"
EMBEDDING_MODEL_PATH = "doc2vec.model"


def get_device() -> torch.device:
    device = (
        acc
        if (acc := torch.accelerator.current_accelerator(check_available=True))
        is not None
        else torch.device("cpu")
    )
    print(f"Using {device.type} device")
    return device


def main():
    # print("--- data crawler start ---")
    # for board_name in BOARD_NAMES:
    #     data_crawler(board_name, REQUIRED_TITLE_COUNT)
    # print("--- data crawler done")
    # print("--- data cleaner start ---")
    # data_cleaner(
    #     [Path(f"{board_name}.csv") for board_name in BOARD_NAMES],
    #     Path(CLEANED_DATA_FILE_PATH),
    # )
    # print("--- data cleaner done ---")
    print("--- data tokenizer start ---")
    data_tokenizer(
        Path(CLEANED_DATA_FILE_PATH),
        Path(TOKENIZED_DATA_FILE_PATH),
        device=get_device(),
    )
    print("--- data tokenizer done ---")
    print("--- data embedding start ---")
    embedding_model = data_embedding(Path(TOKENIZED_DATA_FILE_PATH))
    embedding_model.save(EMBEDDING_MODEL_PATH)
    print("--- data embedding end ---")
    print("--- data classify start ---")
    doc2vec_model: Doc2Vec = Doc2Vec.load(EMBEDDING_MODEL_PATH)  # type: ignore
    doc_vecs = doc2vec_model.dv
    data_classify(Path(TOKENIZED_DATA_FILE_PATH), doc_vecs, get_device())
    print("--- data classify end ---")


if __name__ == "__main__":
    main()
