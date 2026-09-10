import csv
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from config import PATH_CONFIG
from predict import BoardPredictor

predictor = BoardPredictor(
    doc2vec_path=Path(PATH_CONFIG["doc2vec_inference_model_file"]),
    classifier_path=Path(PATH_CONFIG["classifier_model_file"]),
)

app = FastAPI()

templates = Jinja2Templates(directory="templates")


@app.get("/")
def home(request: Request):
    boards = list(predictor.id_to_board.values())
    return templates.TemplateResponse(
        request=request, name="index.html", context={"boards": boards}
    )


@app.get("/api/model/prediction")
def predict(title: str = ""):
    predicted = predictor.predict(title)
    return {"label": predicted}


class FeedbackParams(BaseModel):
    title: str
    label: str


@app.post("/api/model/feedback")
def feedback(params: FeedbackParams):
    with open("user-labeled-titles.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([params.title, params.label])
    return {"message": "OK"}
