import numpy as np
import ncnn
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "models" / "drone_ncnn_model"

with ncnn.Net() as net:
    net.load_param(str(D / "model.ncnn.param"))
    net.load_model(str(D / "model.ncnn.bin"))
    with net.create_extractor() as ex:
        x = np.random.rand(3, 640, 640).astype(np.float32)
        ex.input("in0", ncnn.Mat(x).clone())
        _, out0 = ex.extract("out0")
        print("out0 shape:", np.array(out0).shape)