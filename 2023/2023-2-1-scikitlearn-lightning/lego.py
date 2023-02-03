import lightning as L
from app import SKLearnTraining
from serve_ui import SKLearnServeUI

class MLPipeline(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.training_component = SKLearnTraining()
        self.deployment_component = SKLearnServeUI()

    def run(self, *args, **kwargs) -> None:
        self.training_component.run()
        if self.training_component.has_succeeded:
            self.deployment_component.run()

    def configure_layout(self):
        return {"name": "Serve", "content": self.deployment_component.url}

pipeline = MLPipeline()
app = L.LightningApp(pipeline)
