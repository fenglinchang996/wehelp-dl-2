from pathlib import Path

import torch
from gensim.models import Doc2Vec

from classify import classify as data_classify
from cleaner import cleaner as data_cleaner
from crawler import crawler as data_crawler
from embedding import embedding as data_embedding
from tokenizer import tokenizer as data_tokenizer

PATH_CONFIG = {
    "cleaned_data_file": "cleaned_data.csv",
    "tokenized_data_file": "tokenized_data.csv",
    "doc2vec_model_file": "doc2vec.model",
}

PIPELINE_CONFIG = {
    "run_crawler": True,  # Run web crawler
    "run_cleaner": True,  # Run data cleaner
    "run_tokenizer": True,  # Run CKIP tokenizer
    "run_embedding": True,  # Train Doc2Vec model
    "run_classify": True,  # Train and evaluate classifier
}

GENERAL_CONFIG = {
    "random_seed": 42,
}

CRAWLER_CONFIG = {
    "board_names": [
        "Baseball",
        "Boy-Girl",
        "C_Chat",
        "HatePolitics",
        "Lifeismoney",
        "Military",
        "PC_Shopping",
        "Stock",
        "Tech_Job",
    ],
    "required_title_count": 10000,
}

TOKENIZER_CONFIG = {
    "stop_pos": [
        "Caa",  # 對等連接詞
        "Cab",  # 連接詞：如等等
        "Cba",  # 連接詞：如的話
        "Cbb",  # 關聯連接詞
        "P",  # 介詞
        "T",  # 語助詞
        "I",  # 感嘆詞
        "Di",  # 時態標記
        "Nh",  # 代名詞
        "DE",  # 的之得地
        "SHI",  # 是
    ],  # POS tags to filter out
    "allowed_pos": None,  # Keep only these POS tags if specified
    "min_tokens": 1,  # Minimum tokens per title to retain
    "chunk_size": 4096,  # File read/write chunk size
    "batch_size": 256,  # GPU batch size for CKIP
}

DOC2VEC_CONFIG = {
    "vector_size": 100,  # Dimensionality of document vectors
    "min_count": 2,  # Ignores words with frequency lower than this
    "epochs": 50,  # Number of training epochs
    "dm": 0,  # Architecture: 0: PV-DBOW, 1: PV-DM
    "window": 5,  # Context window size
    "workers": 4,  # Number of worker threads
    "inferring_epochs": 50,  # Epochs for inferring test vectors
    "testing_count": 1000,  # Number of documents sampled for similarity evaluation
    "save_threshold": 0.8,  # Minimum second self-similarity to save model
}

CLASSIFIER_CONFIG = {
    "hidden_layers": [100, 50],  # Hidden layer dimensions (e.g. [100], [50, 32])
    "activation": "relu",  # Activation: 'relu', 'leaky_relu'
    "optimizer_type": "sgd",  # Optimizer: 'sgd', 'adam'
    "learning_rate": 0.01,  # Learning rate
    "epochs": 30,  # Training epochs
    "batch_size": 64,  # DataLoader batch size
    "train_ratio": 0.8,  # Ratio of dataset for training (e.g. 0.8 for 8:2)
}


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
    # 1. Crawler
    if PIPELINE_CONFIG["run_crawler"]:
        print("--- data crawler start ---")
        for board_name in CRAWLER_CONFIG["board_names"]:
            data_crawler(board_name, CRAWLER_CONFIG["required_title_count"])
        print("--- data crawler done ---")

    # 2. Cleaner
    if PIPELINE_CONFIG["run_cleaner"]:
        print("--- data cleaner start ---")
        data_cleaner(
            [Path(f"{board_name}.csv") for board_name in CRAWLER_CONFIG["board_names"]],
            Path(PATH_CONFIG["cleaned_data_file"]),
        )
        print("--- data cleaner done ---")

    # 3. Tokenizer
    if PIPELINE_CONFIG["run_tokenizer"]:
        print("--- data tokenizer start ---")
        data_tokenizer(
            input=Path(PATH_CONFIG["cleaned_data_file"]),
            output=Path(PATH_CONFIG["tokenized_data_file"]),
            stop_pos=TOKENIZER_CONFIG["stop_pos"],
            allowed_pos=TOKENIZER_CONFIG["allowed_pos"],
            min_tokens=TOKENIZER_CONFIG["min_tokens"],
            chunk_size=TOKENIZER_CONFIG["chunk_size"],
            batch_size=TOKENIZER_CONFIG["batch_size"],
            device=get_device(),
        )
        print("--- data tokenizer done ---")

    # 4. Doc2Vec Embedding
    if PIPELINE_CONFIG["run_embedding"]:
        print("--- data embedding start ---")
        data_embedding(
            input_corpus_path=Path(PATH_CONFIG["tokenized_data_file"]),
            vector_size=DOC2VEC_CONFIG["vector_size"],
            min_count=DOC2VEC_CONFIG["min_count"],
            epochs=DOC2VEC_CONFIG["epochs"],
            dm=DOC2VEC_CONFIG["dm"],
            window=DOC2VEC_CONFIG["window"],
            workers=DOC2VEC_CONFIG["workers"],
            inferring_epochs=DOC2VEC_CONFIG["inferring_epochs"],
            testing_count=DOC2VEC_CONFIG["testing_count"],
            save_threshold=DOC2VEC_CONFIG["save_threshold"],
            model_output_path=Path(PATH_CONFIG["doc2vec_model_file"]),
            random_seed=GENERAL_CONFIG["random_seed"],
        )
        print("--- data embedding end ---")

    # 5. Classifier
    if PIPELINE_CONFIG["run_classify"]:
        print("--- data classify start ---")
        doc2vec_model: Doc2Vec = Doc2Vec.load(PATH_CONFIG["doc2vec_model_file"])  # type: ignore
        doc_vecs = doc2vec_model.dv
        data_classify(
            corpus_file=Path(PATH_CONFIG["tokenized_data_file"]),
            vecs=doc_vecs,
            board_names=CRAWLER_CONFIG["board_names"],
            hidden_layers=CLASSIFIER_CONFIG["hidden_layers"],
            activation=CLASSIFIER_CONFIG["activation"],
            optimizer_type=CLASSIFIER_CONFIG["optimizer_type"],
            learning_rate=CLASSIFIER_CONFIG["learning_rate"],
            epochs=CLASSIFIER_CONFIG["epochs"],
            batch_size=CLASSIFIER_CONFIG["batch_size"],
            train_ratio=CLASSIFIER_CONFIG["train_ratio"],
            random_seed=GENERAL_CONFIG["random_seed"],
            device=get_device(),
        )
        print("--- data classify end ---")


if __name__ == "__main__":
    main()
