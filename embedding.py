import csv
import random
from pathlib import Path

from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class EpochLogger(CallbackAny2Vec):

    def __init__(self, total_epoches: int):
        self.epoch = 1
        self._total_epoches = total_epoches

    def on_epoch_begin(self, model):
        pass

    def on_epoch_end(self, model):
        print(f"\rEpoch {self.epoch} finished ", end="", flush=True)
        if self.epoch == self._total_epoches:
            print(f"\r{self._total_epoches} epochs training finished", flush=True)
        self.epoch += 1


def read_corpus(input_corpus: Path) -> list[TaggedDocument]:
    with open(input_corpus, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        tagged_docs: list[TaggedDocument] = []
        for i, row in enumerate(reader):
            tokens = row[1:]
            if not tokens:
                continue
            tagged_docs.append(TaggedDocument(tokens, [i]))
    return tagged_docs


def get_random_docs(docs: list[TaggedDocument], count: int, random_seed: int = 42):
    random.seed(random_seed)
    sample_size = min(count, len(docs))
    return random.sample(docs, sample_size)


def similarity(
    model: Doc2Vec, testing_docs: list[TaggedDocument], inferring_epochs: int = 50
) -> tuple[float, float]:
    sim_ranks = []
    for doc in testing_docs:
        vec = model.infer_vector(doc.words, epochs=inferring_epochs)
        sims = model.dv.most_similar([vec], topn=2)
        top_tags = [tag for tag, _ in sims]
        if doc.tags[0] == top_tags[0]:
            sim_ranks.append(1)
        elif len(top_tags) > 1 and doc.tags[0] == top_tags[1]:
            sim_ranks.append(2)
        else:
            sim_ranks.append(0)
    self_similarity = len(list(filter(lambda x: x == 1, sim_ranks))) / len(sim_ranks)
    second_self_similarity = len(
        list(filter(lambda x: x == 1 or x == 2, sim_ranks))
    ) / len(sim_ranks)
    return self_similarity, second_self_similarity


def embedding(
    input_corpus_path: Path,
    vector_size: int = 100,
    min_count: int = 2,
    epochs: int = 50,
    dm: int = 0,
    window: int = 5,
    workers: int = 6,
    inferring_epochs: int = 50,
    testing_count: int = 1000,
    save_threshold: float = 0.8,
    model_output_path: Path | None = None,
    display_logs: bool = True,
    random_seed: int = 42,
):
    training_docs = read_corpus(input_corpus_path)
    epoch_logger = EpochLogger(total_epoches=epochs)
    model = Doc2Vec(
        vector_size=vector_size,
        dm=dm,
        dbow_words=0,
        window=window,
        min_count=min_count,
        workers=workers,
    )
    model.build_vocab(training_docs)
    print("--- Doc2Vec training start ---")
    model.train(
        training_docs,
        total_examples=model.corpus_count,
        epochs=epochs,
        callbacks=[epoch_logger],
    )
    print("--- Doc2Vec training end ---")
    print("--- Self similarity start ---")
    testing_docs = get_random_docs(
        training_docs, testing_count, random_seed=random_seed
    )
    self_similarity, second_self_similarity = similarity(
        model, testing_docs, inferring_epochs=inferring_epochs
    )
    print(f"Self similarity with {testing_count} docs: {self_similarity:.4f}")
    print(
        f"Second self similarity with {testing_count} docs : {second_self_similarity:.4f}"
    )
    print("--- Self similarity end ---")

    if model_output_path is not None and second_self_similarity >= save_threshold:
        model.save(str(model_output_path))
        print(f"Embedding model saved successfully to {model_output_path}")

    result = {
        "model": model,
        "self_similarity": self_similarity,
        "second_self_similarity": second_self_similarity,
    }
    return result
