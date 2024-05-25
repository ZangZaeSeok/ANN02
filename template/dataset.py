import torch
from torch.utils.data import Dataset
import re

class Shakespeare(Dataset):
    """ Shakespeare dataset

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
           You need this dictionary to generate characters.
        2) Make list of character indices using the dictionary
        3) Split the data into chunks of sequence length 30. 
           You should create targets appropriately.
    """

    def __init__(self, input_file):
        with open(input_file, 'r') as f:
            text = f.read()

        # 1.1 Load input file
            # 특수 문자의 경우, 빈도가 낮으나, 성능이 저하되게 만들 여지가 존재하므로, 제거
            # 대문자와 소문자는 굳이 구분해서 발생되는 정보의 차이가 크게 없기 때문에 모든 단어를 소문자화
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)

        # 1.2 construct character dictionary {index:character}
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.chars)}
        self.idx_to_char = {idx: ch for idx, ch in enumerate(self.chars)}

        # 2. Make list of character indices using the dictionary
        self.data = [self.char_to_idx[ch] for ch in text]

        # 3.1 Split the data into chunks of sequence length 30. 
            ## sequence length를 30으로 세팅
        self.seq_length = 30

        
    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        # 3.2 Split the data into chunks of sequence length 30. 
            ## 설정해둔 sequence_length만큼 input text와 target text 생성
        input_seq = torch.tensor(self.data[idx:idx + self.seq_length], dtype=torch.long)
        target_seq = torch.tensor(self.data[idx + 1:idx + 1 + self.seq_length], dtype=torch.long)
        return input_seq, target_seq

if __name__ == '__main__':
    dataset = Shakespeare('/workspace/ANN/days1/data/shakespeare_train.txt')
    print(f'Dataset size: {len(dataset)}')
    input_seq, target_seq = dataset[0]
    print(f'Input sequence: {input_seq}')
    print(f'Target sequence: {target_seq}')
    print(f'char_to_idx: {dataset.char_to_idx}')
