from BrownWheel.BrownToolBox import toolbox
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm



raw_image_path=r'D:\PycharmProjects\pytorch\cnn_food_classification\food-11\training'
BATCH_SIZE=100
SHUFFLE=True
EPOCH=10
device=torch.device('cuda')
lr=0.01

X,y=toolbox.readimage(path=raw_image_path,hasLabel=True,imreadflag=0,image_size=128)

train_set=Data.TensorDataset(X,y)

train_loader=Data.DataLoader(
    dataset=train_set,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
)
torch.cuda.empty_cache()
network=nn.Sequential(

    nn.Conv2d(1, 64, 3, 1, 1),  # [64, 128, 128]
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2, 2, 0),  # [64, 64, 64]

    nn.Conv2d(64, 128, 3, 1, 1),  # [128, 64, 64]
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(2, 2, 0),  # [128, 32, 32]

    nn.Conv2d(128, 256, 3, 1, 1),  # [256, 32, 32]
    nn.BatchNorm2d(256),
    nn.ReLU(),
    nn.MaxPool2d(2, 2, 0),  # [256, 16, 16]

    nn.Conv2d(256, 512, 3, 1, 1),  # [512, 16, 16]
    nn.BatchNorm2d(512),
    nn.ReLU(),
    nn.MaxPool2d(2, 2, 0),  # [512, 8, 8]

    nn.Conv2d(512, 512, 3, 1, 1),  # [512, 8, 8]
    nn.BatchNorm2d(512),
    nn.ReLU(),
    nn.MaxPool2d(2, 2, 0),  # [512, 4, 4]

    nn.Flatten(start_dim=1),

    nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
).to(device)

optimzer=torch.optim.Adam(network.parameters(), lr=lr)

count=0

for epoch in range(EPOCH):
    count+=1
    print(f'EPOCH {count}')
    total_lost = 0
    correct_count=0
    total_data_amount=0
    for batch in tqdm(train_loader):
        X=batch[0].to(device=device,dtype=torch.float)
        y=batch[1].to(device=device,dtype=torch.long)
        pred=network(X)
        loss=F.cross_entropy(pred,y)

        optimzer.zero_grad()

        loss.backward()
        optimzer.step()

        total_lost+=loss
        correct_count+=toolbox.count_correct(pred,y)
        total_data_amount+=len(y)
    print(f'total_lost {total_lost} tatal sample{total_data_amount} correct_count {round(correct_count,3)} accuracy{round((correct_count/total_data_amount*100),2)}%')



