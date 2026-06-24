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
    return float(outputs[0][0,0]), float(outputs[0][0,1])

project = gr.Interface(
    fn=greet,
    inputs=[gr.Number(label="clinical systolic blood pressure"), gr.Number(label="clinical DBP"),
             gr.Number(label="eGFR"), gr.Number(label="body mass index"), 
             gr.Number(label="nRAAs drug use"), gr.Number(label="history of hypertension"), gr.Number(label="age")],
    outputs=[gr.Number(label="a"), gr.Number(label ="b")],
    api_name="predict"
)
project.launch()
