"""Convierte los .keras (carpeta modelos_keras) a JSON para el navegador. Solo requiere h5py:
    pip install h5py && python scripts/exportar_modelos.py"""
import base64, io, json, zipfile
import h5py, numpy as np

def b64(a): return base64.b64encode(np.asarray(a, dtype="<f4").tobytes()).decode()

def exportar(src, dst):
    z = zipfile.ZipFile(src)
    cfg = json.loads(z.read("config.json"))["config"]["layers"]
    f = h5py.File(io.BytesIO(z.read("model.weights.h5")))["layers"]
    cont, capas = {}, []
    def vars_(clase):
        n = cont.get(clase, 0); cont[clase] = n + 1
        nom = {"Conv2D": "conv2d", "Dense": "dense", "BatchNormalization": "batch_normalization"}[clase]
        g = f[nom if n == 0 else f"{nom}_{n}"]["vars"]
        return [g[str(i)][()] for i in range(len(g))]
    for l in cfg:
        c, k = l["class_name"], l["config"]
        if c == "Conv2D":
            w, b = vars_(c); capas.append({"t": "conv", "ks": list(w.shape), "k": b64(w), "b": b64(b)})
        elif c == "BatchNormalization":
            g, be, mu, var = vars_(c)
            capas.append({"t": "bn", "g": b64(g), "be": b64(be), "mu": b64(mu), "var": b64(var), "eps": k["epsilon"]})
        elif c == "MaxPooling2D": capas.append({"t": "pool"})
        elif c == "Flatten": capas.append({"t": "flatten"})
        elif c == "Dense":
            w, b = vars_(c); capas.append({"t": "dense", "ks": list(w.shape), "w": b64(w), "b": b64(b), "act": k["activation"]})
    json.dump({"capas": capas}, open(dst, "w"))
    print(src, "->", dst)

exportar("modelos_keras/modelo_mejorado.keras", "public/modelos/cnn.json")
exportar("modelos_keras/modelo_baseline.keras", "public/modelos/mlp.json")
