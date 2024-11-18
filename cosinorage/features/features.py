from ..dataloaders import DataLoader

# TODO: Implement the WearableFeatures class
class WearableFeatures:

    def __init__(self, loader: DataLoader):

        self.enmo = loader.get_enmo_minute()
        
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
        pass
    
    def compute_IS(self):
        pass

    def compute_RA(self):
        pass

    def compute_M10(self):
        pass
    
    def compute_M5(self):
        pass

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
