from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from model.model import LLM
import torch

app = FastAPI()

class InputText(BaseModel):
    text: str

# "bigscience/bloomz-1b1"
model_tag = "bigscience/bloomz-1b1"
model = LLM(model_name = model_tag,
            device = "cuda" if torch.cuda.is_available() else "cpu")

@app.get("/")
async def docs_redirect():
    return RedirectResponse(url='/docs')

@app.post("/language-detection")
def language_detection(text):
    return {"language": model.language_detection(text)}

@app.post("/entity-recognition")
def ner(text):
    return model.entity_recognition(text)