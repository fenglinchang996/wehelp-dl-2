from pathlib import Path

import torch
from ckip_transformers.nlp import CkipPosTagger, CkipWordSegmenter
from gensim.models import Doc2Vec

from config import PATH_CONFIG
from model import NeuralNetwork
from tokenizer import is_stop_pos

default_device = torch.device("cpu")


class BoardPredictor:
    def __init__(
        self,
        doc2vec_path: Path,
        classifier_path: Path,
        device: torch.device = default_device,
    ) -> None:
        self.device = device

        self.doc2vec: Doc2Vec = Doc2Vec.load(str(doc2vec_path))  # type: ignore

        classifier_info = torch.load(str(classifier_path))
        self.id_to_board = classifier_info["id_to_board"]
        self.classifier = NeuralNetwork(
            input_dim=classifier_info["input_dim"],
            hidden_layers=classifier_info["hidden_layers"],
            class_num=classifier_info["class_num"],
            activation=classifier_info["activation"],
        ).to(self.device)
        self.classifier.load_state_dict(classifier_info["state_dict"])
        self.classifier.eval()

        self.ws_driver = CkipWordSegmenter(model="bert-base", device=self.device)
        self.pos_driver = CkipPosTagger(model="bert-base", device=self.device)

    def predict(self, title: str):
        ws_result = self.ws_driver([title.strip().lower()])
        pos_result = self.pos_driver(ws_result)
        tokens: list[str] = [
            w.strip()
            for w, p in zip(ws_result[0], pos_result[0])
            if not is_stop_pos(p) and w.strip()
        ]

        if not tokens:
            return None

        vector = self.doc2vec.infer_vector(tokens, epochs=50)
        x = torch.tensor(vector, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.classifier(x)
            predicted = torch.argmax(outputs)
        predicted_board = self.id_to_board[predicted.item()]
        return predicted_board


def main():
    predictor = BoardPredictor(
        doc2vec_path=Path(PATH_CONFIG["doc2vec_inference_model_file"]),
        classifier_path=Path(PATH_CONFIG["classifier_model_file"]),
    )

    title = "[請益] 預算 40K 遊戲機 4070S 顯卡推薦"
    result = predictor.predict(title)
    print(result)


if __name__ == "__main__":
    main()
