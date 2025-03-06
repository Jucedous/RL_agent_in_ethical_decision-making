from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from stable_baselines3 import PPO  
from transformers import BertTokenizer
import torch

import numpy as np
bert_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


MODEL_PATH = "ppo_ethical_decision_model.zip"  
model = PPO.load(MODEL_PATH)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/", response_class=HTMLResponse)
async def predict(request: Request, input_data: str = Form(...)):
    encoding = tokenizer(
            input_data, return_tensors='pt', padding='max_length', truncation=True, max_length=128
        )
    state = torch.cat([
            encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze()
        ])
    action, _states = model.predict(state.numpy())
    if action == 1:
        result = "not ethical"
    else:
        result = "ethical"
    print(result)

    return templates.TemplateResponse("index.html", {"request": request, "result": result})