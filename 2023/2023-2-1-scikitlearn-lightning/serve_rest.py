# !python -m pip install -U scikit-learn fastapi
import lightning as L
from joblib import load
from lightning.app.components.serve import PythonServer
from lightning.app.storage import Drive
from pydantic import BaseModel

FEATURE_NAMES = ["Setosa", "Versicolor", "Virginica"]


class FeaturesInput(BaseModel):
    sepal_width: float
    sepal_length: float
    petal_width: float
    petal_length: float


class ModelServe(PythonServer):
    def __init__(self):
        super().__init__(input_type=FeaturesInput)
        # cloud persistable storage for model checkpoint
        self.model_storage = Drive("lit://checkpoints")

    def setup(self):
        self.model_storage.get("model.joblib")
        self._model = load("model.joblib")

    def predict(self, request: FeaturesInput):
        sepal_width = request.sepal_width
        sepal_length = request.sepal_length
        petal_width = request.petal_width
        petal_length = request.petal_length

        class_idx = self._model.predict(
            [[sepal_width, sepal_length, petal_width, petal_length]]
        )[0]
        print(class_idx)
        return {"prediction": FEATURE_NAMES[class_idx]}


# 1. Launch REST API
component = ModelServe()
app = L.LightningApp(component)


# 2. AutoScale the API
# from lightning.app.components import AutoScaler

# scalable_component = AutoScaler(ModelServe, max_replicas=2, input_type=FeaturesInput)
# app = L.LightningApp(scalable_component)
