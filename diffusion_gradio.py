import onnxruntime as ort
import gradio as gr
import numpy as np
from scipy.special import softmax

sess = ort.InferenceSession("onnx_diffusion.onnx")

def greet(p1, p2, p3 , p4, p5, p6, p7):
    outputs = sess.run(
    None,
    {
        "cond": np.array([[p1, p2, p3, p4, p5, p6, p7]]).astype(np.float32),
        "t": np.array([500], dtype=np.float32),
    }
)
    output_fake= outputs[0]
    output_fake = softmax(output_fake, axis = 1)
    outputs = output_fake[:,1]
    outputs = outputs[0]
    if (outputs > 0.692):
        risk = "High Risk"
    elif (outputs > 0.515):
        risk = "Medium Risk"
    else:
        risk = "Low Risk"
    return outputs * 100, risk

project = gr.Interface(
    fn=greet,
    inputs=[gr.Number(label="Clinical Systolic Blood Pressure"), gr.Number(label="Clinical DBP"),
             gr.Number(label="eGFR"), gr.Number(label="Body Mass Index"), 
             gr.Number(label="nRAAs Drug Use"), gr.Number(label="History of Hypertension"), gr.Number(label="Age")],
    outputs=[gr.Number(label="NH Prediction (%)"), gr.Textbox(label="Patient Risk")],
    api_name="predict"
)
project.launch(share=False)