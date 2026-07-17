import onnxruntime as ort
import gradio as gr
import numpy as np
from scipy.special import softmax

sess = ort.InferenceSession("onnx_diffusion.onnx")

def greet(p1, p2, p3 , p4, p5, p6, p7):
    #define outputs from the model
    outputs = sess.run(
    None,
    {
        #7 selected varibles as input array
        "cond": np.array([[p1, p2, p3, p4, p5, p6, p7]]).astype(np.float32),
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
    outputs = float(outputs[0])
    #classify the patient risk based on calculated output percentage
    if (outputs > 0.692):
        risk = "High Risk"
    elif (outputs > 0.515):
        risk = "Medium Risk"
    else:
        risk = "Low Risk"
    #return the NH percentage and risk classification
    return int(round(outputs * 100)), risk

project = gr.Interface(
    fn=greet,
    inputs=[gr.Number(label="Clinical Systolic Blood Pressure (mmHg)", minimum= 0, maximum = 300), gr.Number(label="Clinical DBP (mmHg)", minimum =0, maximum = 300),
             gr.Number(label="eGFR (ml/min/1.73m^2)", minimum =0, maximum = 200), gr.Number(label="Body Mass Index (kg/m^2)", minimum = 0, maximum = 50), 
             gr.Number(label="nRAAs Drug Use", minimum = 0, maximum = 1), gr.Number(label="History of Hypertension", minimum = 0, maximum = 1), gr.Number(label="Age (years)", minimum= 0, maximum = 100)],
    outputs=[gr.Number(label="NH Prediction (%)"), gr.Textbox(label="Patient Risk")],
    api_name="predict"
)
project.launch(share=False)