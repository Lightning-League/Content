# !pip install ultralytics lightning-api-access
# !pip uninstall -y opencv-python opencv-python-headless
# !pip install opencv-python-headless==4.5.5.64 

import lightning as L
from lightning.app.components.serve import PythonServer
from pydantic import BaseModel

class InputType(BaseModel):
    url: str

class Detections(BaseModel):
    prediction: list

class YoloV8Server(PythonServer):
    def setup(self):
        from ultralytics import YOLO
        self._model = YOLO("yolov8n.pt")

    def predict(self, request: InputType):
        preds = self._model.predict(request.url)[0].tolist()
        return {"prediction": preds}

component = YoloV8Server(input_type=InputType, output_type=Detections)
app = L.LightningApp(component)
