# !pip install ultralytics lightning-api-access
# !pip uninstall -y opencv-python opencv-python-headless
# !pip install opencv-python-headless==4.5.5.64 

import lightning as L
from lightning.app.components.serve import PythonServer
from lightning.app.components import AutoScaler
from pydantic import BaseModel

class InputType(BaseModel):
    image_url: str

class Detections(BaseModel):
    prediction: list

class YoloV8Server(PythonServer):
    def setup(self):
        from ultralytics import YOLO
        self._model = YOLO("yolov8n.pt")

    def predict(self, request: InputType):
        preds = self._model.predict(request.image_url)
        print(preds)
        classes = preds[0].boxes.cls
        results = [self._model.names[int(cls)] for cls in classes]
        # boxes
        return {"prediction": results}

class RootFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.component = YoloV8Server(input_type=InputType, output_type=Detections, port = 8888)
    
    def run(self):
        self.component.run()
        if self.component.url:
            print(self.component.url)


app = L.LightningApp(RootFlow())
