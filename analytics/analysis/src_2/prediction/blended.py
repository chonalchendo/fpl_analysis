from analysis.src_2.models.blend import BlendedRegressor
from analysis.src_2.preprocessing.pipeline.build import PipelineBuilder

class BlendedPredictor(PipelineBuilder):
    def __init__(self, drop_features: list[str], target_encode_features: list[str]) -> None:
        super().__init__(drop_features, target_encode_features)
        
    # def _preprocess(self) -> BlendedRegressor:
    #     self.pipes_ = [self.build(model=model) for model in self.models]
    
    # take in X and y
    # clean X and y
    # fit X and y to models
    # predict X with weights
    # return prediction