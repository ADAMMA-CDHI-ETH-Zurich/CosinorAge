from ..dataloaders import DataLoader

from .utils.nonparam_analysis import *

# TODO: Implement the WearableFeatures class
class WearableFeatures:

    def __init__(self, loader: DataLoader):

        self.enmo = loader.get_enmo_data()
        
        # cosinor analysis features

        # circadian analysis features
        self.IV = None
        self.IS = None
        self.RA = None
        self.M10 = None
        self.M5 = None
        self.M10_start = None
        self.M5_start = None

        # physical activity metrics
        self.SB = None
        self.LIPA = None
        self.MVPA = None

        # sleep metrics
        self.TST = None
        self.WASO = None
        self.sleep_regularity = None
        self.sleep_efficiency = None
    

    def compute_IV(self):
        self.IV = IV(self.enmo)
    
    def compute_IS(self):
        self.IS = IS(self.enmo)

    def compute_RA(self):
        self.RA = RA(self.enmo) 

    def compute_M10(self):
        self.M10 = M10(self.enmo)
    
    def compute_M5(self):
        self.M5 = M5(self.enmo)

    def compute_M10_start(self):
        pass
    
    def compute_M5_start(self):
        pass

    def compute_SB(self):
        pass
    
    def compute_LIPA(self):
        pass
    
    def compute_MVPA(self):
        pass

    def compute_TST(self):
        pass
    
    def compute_WASO(self):
        pass

    def compute_sleep_regularity(self):
        pass
    
    def compute_sleep_efficiency(self):
        pass

    def get_IV(self):
        return self.IV
    
    def get_IS(self):
        return self.IS
    
    def get_RA(self):
        return self.RA
    
    def get_M10(self):
        return self.M10
    
    def get_M5(self):
        return self.M5
