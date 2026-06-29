import onnxruntime as ort
import gradio as gr
import numpy as np

sess = ort.InferenceSession("onnx_diffusion.onnx")

def greet(p1, p2, p3 , p4, p5, p6, p7):
    outputs = sess.run(
    None,
    {
        "cond": np.array([[p1, p2, p3, p4, p5, p6, p7]]).astype(np.float32),
        "t": np.array([500], dtype=np.float32),
    }
)
    return outputs[0][0,0]

project = gr.Interface(
    fn=greet,
    inputs=[gr.Number(label="Clinical Systolic Blood Pressure"), gr.Number(label="Clinical DBP"),
             gr.Number(label="eGFR"), gr.Number(label="Body Mass Index"), 
             gr.Number(label="nRAAs Drug Use"), gr.Number(label="History of Hypertension"), gr.Number(label="Age")],
    outputs=[gr.Number(label="NH Prediction")],
    api_name="predict"
)
project.launch(share=False)