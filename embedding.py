import csv
import random
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.models.callbacks import CallbackAny2Vec
from pathlib import Path

DOC_2_VEC_VECTOR_SIZE = 200
DOC_2_VEC_EPOCHS = 75
TESTING_DOC_COUNT = 5000


class EpochLogger(CallbackAny2Vec):

    def __init__(self):
        self.epoch = 1

    def on_epoch_begin(self, model):
        pass

    def on_epoch_end(self, model):
        print(f"Epoch {self.epoch} finished")
        self.epoch += 1


def read_corpus(input_file: Path):
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        tagged_docs: list[TaggedDocument] = []
        for i, row in enumerate(reader):
            tokens = row[1:]
            if not tokens:
                continue
            tagged_docs.append(TaggedDocument(tokens, [i]))
    return tagged_docs


def get_random_docs(docs: list[TaggedDocument], count: int):
    random.seed(42)
    sample_size = min(count, len(docs))
    return random.sample(docs, sample_size)


def similarity(model: Doc2Vec, testing_docs: list[TaggedDocument]):
    sim_ranks = []
    for doc in testing_docs:
        vec = model.infer_vector(doc.words)
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


def embedding(input_file: Path):
    training_docs = read_corpus(input_file)
    epoch_logger = EpochLogger()
    model = Doc2Vec(vector_size=DOC_2_VEC_VECTOR_SIZE, min_count=2, workers=8)
    model.build_vocab(training_docs)
    print("--- model training start ---")
    model.train(
        training_docs,
        total_examples=model.corpus_count,
        epochs=DOC_2_VEC_EPOCHS,
        callbacks=[epoch_logger],
    )
    print("--- model training end ---")
    print("--- self similarity start ---")
    testing_docs = get_random_docs(training_docs, TESTING_DOC_COUNT)
    self_similarity, second_self_similarity = similarity(model, testing_docs)
    print(f"self similarity with {TESTING_DOC_COUNT} docs: {self_similarity}")
    print(
        f"second_self_similarity with {TESTING_DOC_COUNT} docs : {second_self_similarity}"
    )
    print("--- self similarity end ---")
    if second_self_similarity > 0.8:
        model.save("doc2vec_model")
