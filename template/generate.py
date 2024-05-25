import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from dataset import Shakespeare
from model import CharRNN, CharLSTM

# -> down_temperature -> temperature의 분산을 점진적으로 줄이면서 문장을 생성 
# -> up_temperature -> temperature의 분산을 점진적으로 높이면서 문장을 생성 

def generate(model, seed_characters, temperature=None, down_temperature=False, up_temperature=False):
    """ Generate characters

    Args:
        model: trained model
        seed_characters: seed characters
        temperature: T
        
    args: other arguments if needed

    Returns:
        samples: generated characters
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if down_temperature:
        generated_text = model.generate_down(seed_characters, length=100, device=device)

        return generated_text
    
    elif up_temperature:
        generated_text = model.generate_up(seed_characters, length=100, device=device)

        return generated_text
        
    generated_text = model.generate(seed_characters, temperature, length=100, device=device)
    
    return generated_text