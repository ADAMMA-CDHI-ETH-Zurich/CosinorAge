from ..dataloaders import DataLoader
from ..features.features import WearableFeatures

# TODO: Implement Bioages class
class CosinorAge:
    
    def __init__(self, features: WearableFeatures, model=None):
        
        # patient data
        self.mesor = features.feature_df.loc["MESOR"]
        self.amplitude = features.feature_df.loc["amplitude"]
        self.acrophase = features.feature_df.loc["acrophase"]
        self.chronological_age = None

        # model parameters
        self.fit = False
        self.model = None
        self.beta_1 = None
        self.beta_2 = None
        self.beta_3 = None
        self.alpha = None

        if model is not None:
            self.fit = True
            self.model = model

    def fit(self):
        pass

    def predict(self):
        adjustment = ((self.beta_1 * self.mesor) + (self.beta_2 * self.amplitude) + (self.beta_3 * self.acrophase))/self.alpha
        cosinor_age = self.chronological_age - adjustment
        return cosinor_age
    