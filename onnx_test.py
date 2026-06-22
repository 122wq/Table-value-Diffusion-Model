import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("onnx_diffusion.onnx")

print([i.name for i in sess.get_inputs()])
print([o.name for o in sess.get_outputs()])

outputs = sess.run(
    None,
    {
        "cond": np.random.randn(1,7).astype(np.float32),
        "t": np.array([10], dtype=np.float32),
    }
)

print(outputs)