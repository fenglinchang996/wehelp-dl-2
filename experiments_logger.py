import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from gensim.models import Doc2Vec


def count_csv_rows(file_path: Path) -> int | None:
    if not file_path.exists():
        return None
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for line in f if line.strip())


def extract_doc2vec_info(model: Doc2Vec, embedding_result: dict | None = None) -> dict:
    return {
        "vector_size": model.vector_size,
        "min_count": getattr(model, "min_count", None),
        "epochs": model.epochs,
        "dm": "PV-DM" if model.dm else "PV-DBOW",
        "window": getattr(model, "window", None),
        "workers": getattr(model, "workers", None),
        "training_docs_count": getattr(model, "corpus_count", len(model.dv)),
        "self_similarity": (
            embedding_result.get("self_similarity") if embedding_result else None
        ),
        "second_self_similarity": (
            embedding_result.get("second_self_similarity") if embedding_result else None
        ),
    }


def log(
    pipeline_config: dict,
    path_config: dict,
    crawler_config: dict,
    tokenizer_config: dict,
    doc2vec_config: dict,
    classifier_config: dict,
    embedding_result: dict | None,
    classifier_result: dict | None,
    log_file_path: Path = Path("results.jsonl"),
):
    timestamp = datetime.now(tz=timezone(timedelta(hours=8))).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # Datasets
    cleaned_file = Path(path_config["cleaned_data_file"])
    tokenized_file = Path(path_config["tokenized_data_file"])
    source_count = count_csv_rows(cleaned_file)
    tokenized_count = count_csv_rows(tokenized_file)

    # Embedding
    model: Doc2Vec | None = None
    if embedding_result and "model" in embedding_result:
        model = embedding_result["model"]
    elif Path(path_config["doc2vec_model_file"]).exists():
        try:
            model = Doc2Vec.load(str(path_config["doc2vec_model_file"]))  # type: ignore
        except Exception:
            model = None

    doc2vec_info = (
        extract_doc2vec_info(model, embedding_result) if model is not None else {}
    )

    # Classifier
    input_dim = doc2vec_info.get("vector_size") or doc2vec_config.get("vector_size")
    class_num = len(crawler_config.get("board_names", []))
    nn_layer_str = f"{input_dim}x{'x'.join(map(str,classifier_config['hidden_layers']))}x{class_num}"
    classifier_info = {}
    if pipeline_config.get("classify"):
        classifier_info = {
            "network_layer": nn_layer_str,
            "activation": classifier_config["activation"],
            "optimizer": classifier_config["optimizer_type"],
            "learning_rate": classifier_config["learning_rate"],
            "epochs": classifier_config["epochs"],
            "batch_size": classifier_config["batch_size"],
            "training_docs_count": (
                classifier_result.get("train_count") if classifier_result else None
            ),
            "testing_docs_count": (
                classifier_result.get("test_count") if classifier_result else None
            ),
            "testing_accuracy": (
                classifier_result.get("testing_accuracy") if classifier_result else None
            ),
        }

    record = {
        "timestamp": timestamp,
        "executed_steps": [k for k, v in pipeline_config.items() if v],
        "dataset": {
            "source_titles_count": source_count,
            "tokenized_titles_count": tokenized_count,
        },
        "tokenizer": {
            "min_tokens": tokenizer_config.get("min_tokens"),
            "removed_pos": tokenizer_config.get("removed_pos"),
            "allowed_pos": tokenizer_config.get("allowed_pos"),
        },
        "doc2vec": doc2vec_info,
        "classifier": classifier_info,
    }

    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n" + "=" * 65)
    print(f"RUN SUMMARY ({timestamp})")
    executed_str = (
        ", ".join(record["executed_steps"]) if record["executed_steps"] else "None"
    )
    print(f"Executed:    {executed_str}")
    print(
        f"[Dataset]    Source Titles: {source_count} | Tokenized Titles: {tokenized_count}"
    )

    tok = record["tokenizer"]
    pos_desc = (
        f"allowed_pos: {tok['allowed_pos']}"
        if tok.get("allowed_pos")
        else f"removed_pos: {len(tok.get('removed_pos') or [])} tags"
    )
    print(f"[Tokenizer]  min_tokens: {tok.get('min_tokens')} | {pos_desc}")

    if doc2vec_info:
        dm_str = doc2vec_info.get("dm", "")
        sim_str = (
            f"Self-Sim: {doc2vec_info['self_similarity'] * 100:.2f}%"
            if doc2vec_info.get("self_similarity")
            else "N/A"
        )
        second_sim_str = (
            f"2nd-Self-Sim: {doc2vec_info['second_self_similarity'] * 100:.2f}%"
            if doc2vec_info.get("second_self_similarity")
            else "N/A"
        )
        print(
            f"[Doc2Vec]    dim: {doc2vec_info.get('vector_size')} | min_count: {doc2vec_info.get('min_count')} | window: {doc2vec_info.get('window')} | workers: {doc2vec_info.get('workers')} | epochs: {doc2vec_info.get('epochs')} | {dm_str}"
        )
        print(
            f"             train_docs: {doc2vec_info.get('training_docs_count')} | {sim_str} | {second_sim_str}"
        )

    if classifier_info:
        acc_str = (
            f"{classifier_info['testing_accuracy'] * 100:.2f}%"
            if classifier_info.get("testing_accuracy") is not None
            else "N/A"
        )
        print(
            f"[Classifier] Network: {classifier_info.get('network_layer')} | act: {classifier_info.get('activation')} | opt: {classifier_info.get('optimizer')} (lr={classifier_info.get('learning_rate')}) | batch: {classifier_info.get('batch_size')} | epochs: {classifier_info.get('epochs')}"
        )
        print(
            f"             train_docs: {classifier_info.get('training_docs_count')} | test_docs: {classifier_info.get('testing_docs_count')}"
        )
        print(f"[Result]     Testing Accuracy: {acc_str}")

    print(f"Saved to {log_file_path}")
    print("=" * 65 + "\n")
