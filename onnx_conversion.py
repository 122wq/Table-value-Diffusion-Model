import onnxruntime as ort
import gradio as gr
import numpy as np

sess = ort.InferenceSession("onnx_diffusion.onnx")

def predict(sbp, dbp, egfr, bmi, nraas, htn, age):

    cond = np.array(
        [[sbp, dbp, egfr, bmi, nraas, htn, age]],
        dtype=np.float32
    )

    outputs = sess.run(
        None,
        {
            "cond": cond,
            "t": np.array([500], dtype=np.float32),
        }
    )

    pred = outputs[0][0]

    return float(pred[0]), float(pred[1])


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Clinical SBP"),
        gr.Number(label="Clinical DBP"),
        gr.Number(label="eGFR"),
        gr.Number(label="BMI"),
        gr.Number(label="nRAAS Drug Use"),
        gr.Number(label="History of Hypertension"),
        gr.Number(label="Age"),
    ],
    outputs=[
        gr.Number(label="Output 1"),
        gr.Number(label="Output 2"),
    ],
    title="Diffusion Model Predictor"
)

demo.launch()