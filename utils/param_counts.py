from models.soccermap import soccermap_model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())






if __name__ == '__main__':
    sm_model = soccermap_model(14,32)
    print("Soccer Map Model Medium parameters:", count_parameters(sm_model))

