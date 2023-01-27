# !python -m pip install -U scikit-learn
import gradio as gr
import lightning as L
from lightning.app.storage import Drive
from lightning.app.components.serve import ServeGradio, PythonServer

from joblib import load

FEATURE_NAMES = ['setosa' 'versicolor' 'virginica']


class SKLearnServe(ServeGradio):
    inputs = [gr.Number(), gr.Number(), gr.Number(), gr.Number()]
    outputs = gr.Text()

    def __init__(self):
        super().__init__(self)
        # cloud persistable storage for model checkpoint
        self.model_storage = Drive("lit://checkpoints")

    def build_model(self):
        self.model_storage.get("model.joblib")
        return load("model.joblib")

    def predict(self, sepal_width, sepal_length, petal_width, petal_length):
        class_idx =  self.model.predict([[sepal_width, sepal_length, petal_width, petal_length]])[0]
        print(class_idx)
        print(FEATURE_NAMES)
        print(FEATURE_NAMES[class_idx])
        return FEATURE_NAMES[class_idx]

component = SKLearnServe()
app = L.LightningApp(component)
