from transformers import RobertaForSequenceClassification, AutoTokenizer
import gdown
from fastapi import FastAPI, Form
import torch
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated



app = FastAPI()

app.add_middleware( #parametros pra liberar a conexao
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


try:

    model = RobertaForSequenceClassification.from_pretrained("./checkpoint/")

    tokenizer = AutoTokenizer.from_pretrained("./checkpoint/", use_fast=False)

except:
    gdown.download_folder(url='https://drive.google.com/drive/folders/1gv9KHJtD5XjneEP6-XHZc1rmeybJioJq', 
                          output='./')
    model = RobertaForSequenceClassification.from_pretrained("./checkpoint/")

    tokenizer = AutoTokenizer.from_pretrained("./checkpoint/", use_fast=False)

@app.post("/predict_sentence")
def predict_sentence(sentence: Annotated[str, Form()]):
    encoded_input = tokenizer(sentence, return_tensors='pt')

    # Dự đoán 
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    predicted_class_idx = scores.argmax()

    # Lấy ra nhãn của class được dự đoán
    labels = ['negative', 'positive', "normal"]  
    predicted_label = labels[predicted_class_idx]

    return {
        "result": predicted_label
    }

@app.post("/predict_sentences")
def predict_sentences(sentences: Annotated[list[str], Form()]):
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    # Dự đoán 
    outputs = model(**inputs)
    scores = outputs[0].detach().cpu().numpy()
    predictions = scores.argmax(axis=-1)
    print(predictions)
    # Lấy ra nhãn tương ứng với mỗi câu 
    labels = ['negative', 'positive', "normal"]
    preds_label = [labels[p] for p in predictions]

    return {
        "result": preds_label
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)