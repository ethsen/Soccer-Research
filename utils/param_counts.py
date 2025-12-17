from models.soccermap import soccermap_model
from models.bettermap import BetterSoccerMap2Head
from models.footballmap import PassMap
from models.pitchvision import PitchVisionNet

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())






if __name__ == '__main__':
    sm_model = PassMap(in_channels=18,base=64,blocks_per_stage=4)
    pvnet = PitchVisionNet(in_channels=18,base=64, blocks_per_stage=3)

    print("Soccer Map Model Medium parameters:", count_parameters(sm_model))
    print("Pitch Vision Net Model Medium parameters:", count_parameters(pvnet))

