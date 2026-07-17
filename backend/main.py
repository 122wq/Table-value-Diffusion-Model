import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy.special import softmax

app = FastAPI(title="Diffusion Model Risk Predictor")

# --- Enable CORS so your Flutter app is allowed to fetch data ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (change to specific domains in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ONNX session
try:
    sess = ort.InferenceSession("onnx_diffusion.onnx")
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    sess = None

class PredictionData(BaseModel):
    cfbg: float
    cDBP: float
    eGFR: float
    bmi: float
    nraas_drug_use: float
    hypertension_history: float
    # Enforces the 0-100 age constraint directly at the API gateway level:
    age: float = Field(..., ge=0, le=100, description="Age must be between 0 and 100") 


@app.post("/predict")
async def predict_data(data: PredictionData):
    if sess is None:
        raise HTTPException(status_code=500, detail="ONNX model is not loaded on the server.")
        
    try:
        # 7 selected variables as input array
        cond_input = np.array([[
            data.cfbg, 
            data.cDBP, 
            data.eGFR, 
            data.bmi, 
            data.nraas_drug_use, 
            data.hypertension_history, 
            data.age
        ]]).astype(np.float32)

        # Run ONNX inference
        outputs = sess.run(
            None,
            {
                "cond": cond_input,
                "t": np.array([500], dtype=np.float32),  # diffusion timestep
            }
        )
        
        # Raw logit processing
        output_fake = outputs[0]
        output_fake = softmax(output_fake, axis=1)
        
        # Convert to a single layer array and extract scalar
        val_numpy = output_fake[0, 1]
        
        # FIX: Safe conversion from numpy.float32 to native Python float
        output_val = float(val_numpy)

        # Classify the patient risk
        if output_val > 0.692:
            risk = "High Risk"
        elif output_val > 0.515:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"

        # Return the NH percentage (as standard python int) and risk classification
        return {
            "prediction_percentage": int(round(output_val * 100)),
            "risk_level": risk
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")