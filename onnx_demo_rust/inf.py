from pytorch_forecasting import TemporalFusionTransformer

class ModelLoader:
    def __init__(self, model_path: str = "tft.ckpt"):
        self.model_path = model_path

    def load(self) -> TemporalFusionTransformer:
        tft = TemporalFusionTransformer.load_from_checkpoint(self.model_path)
        return tft

