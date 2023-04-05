from data_loader import DataLoader
from preprocessing import Preprocessor
from train import Model
from argparse import ArgumentParser
class Pipeline:
    def __init__(self):
        self.data_loader = DataLoader
        self.preprocessor = Preprocessor
        self.model = Model
    
    def run(self):
        data = self.data_loader().run()
        data_modules = self.preprocessor(data).run()
        self.model(data_modules).run()
        return None
    
    def inference(self, data_path: str):
        data = self.data_loader(data_path).run()
        data_modules = self.preprocessor(data).run()
        self.model(data_modules).inference()
        return None

    
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    args = parser.parse_args()
    pipeline = Pipeline()
    if args.data_path:
        pipeline.inference(args.data_path)
    else:
        pipeline.run()