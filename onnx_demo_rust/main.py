from data_loader import DataLoader
from preprocessing import Preprocessor
from train import Model
from inf import ModelLoader
from matplotlib import pyplot as plt
from argparse import ArgumentParser
class Pipeline:
    def __init__(self):
        self.data_loader = DataLoader
        self.preprocessor = Preprocessor
        self.model = Model
        self.model_loader = ModelLoader
    
    def fit(self):
        data = self.data_loader().run_fit()
        data_modules = self.preprocessor(data).run()
        self.model(data_modules).run()
        return None
    
    def inference(self, model_path: str):
        data = self.data_loader().run_inference()
        model = self.model_loader(model_path).load()
        new_raw_predictions, new_x = model.predict(data, mode="raw", return_x=True)
        for idx in range(10):  # plot 10 examples
            model.plot_prediction(
                new_x, new_raw_predictions, idx=idx, show_future_observed=False
                )
            plt.savefig(f"example_{idx}.png")
        return None

    
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()
    pipeline = Pipeline()
    if args.model_path:
        pipeline.inference(args.model_path)
    else:
        pipeline.fit()