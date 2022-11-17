import torch

from torch.utils.data import Dataset
import sys
sys.path.append('/home/yoonbyung/Dev/Real2Sim/src')
from utils.dataloader import load

class SoRoSimDataset(Dataset): 
  def __init__(self,
                data_dir,
                train:bool=True):
    if train:
      data_path = data_dir + '/trainInterpolate.json' #'/home/yoonbyung/Dev/Real2Sim/data/trainInterpolate.json'
    else:
      data_path = data_dir + '/valExtrapolate.json'#'/home/yoonbyung/Dev/Real2Sim/data/valExtrapolate.json'

    x, y = load(data_path)
    self.x_data = x
    self.y_data = y

  def __len__(self): 
    return len(self.x_data)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self, idx): 
    x = torch.FloatTensor(self.x_data[idx])
    y = torch.FloatTensor(self.y_data[idx])
    return x, y

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = SoRoSimDataset(train=False)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
    # print(len(dataloader))
    for samples in dataloader:
        x_train, y_train = samples
        print(x_train.shape, y_train.shape)
