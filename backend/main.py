import onnxruntime as ort
import numpy as np
from scipy.special import softmax
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
sess = ort.InferenceSession("onnx_diffusion.onnx")

class PredictionData(BaseModel):
    cfbg: float
    cDBP: float
    eGFR: float
    bmi: float
    nraas_drug_use: float
    hypertension_history: float
    age: float 


@app.post("/predict")
async def predict_data(data: PredictionData):
    outputs = sess.run(
    None,
    {
        #7 selected varibles as input array
        "cond": np.array([[data.cfbg, 
                           data.cDBP, 
                           data.eGFR, 
                           data.bmi, 
                           data.nraas_drug_use, 
                           data.hypertension_history, 
                           data.age]]).astype(np.float32),
        #diffusion timestep
        "t": np.array([500], dtype=np.float32),
    }
    )
    #raw logit
    output_fake= outputs[0]
    #calculated output
    output_fake = softmax(output_fake, axis = 1)
    #convert to a single layer array
    outputs = output_fake[:,1]
    outputs = outputs[0]
    #classify the patient risk based on calculated output percentage
    if (outputs > 0.692):
        risk = "High Risk"
    elif (outputs > 0.515):
        risk = "Medium Risk"
    else:
        risk = "Low Risk"
    #return the NH percentage and risk classification
    return int(outputs * 100), risk
