#uvicorn mlapi:app --reload
#localhost:8000/docs for swaggerui
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

MODEL = tf.keras.models.load_model('model/')

app = FastAPI()

class UserInput(BaseModel):
    # User_insput of type list of floats with length : samples * 4 : [timestamp , x-axis, y-axis , z-axis , timestamp , x-axis ... ]
    user_input: list

@app.get('/')
async def index():
    return {"Message": "This is Index"}

with open('encoder.pkl', 'rb') as f:
    le = pickle.load(f) 

@app.post('/predict/')
async def predict(UserInput: UserInput):
    
    input_shape = MODEL.layers[0].input_shape
    
    Y = np.array(UserInput.user_input).reshape(1,input_shape[1], input_shape[2])
    
    preds = MODEL.predict(Y).argmax()
    
    list(le.inverse_transform([preds]))

    return {'Prediction' : list(le.inverse_transform([preds]))}


# Get the input shape for the model layer
#input_shape = MODEL.layers[0].input_shape
#input_shape
#model_input = np.array(pil_image).reshape((input_shape[1], input_shape[2], input_shape[3]))