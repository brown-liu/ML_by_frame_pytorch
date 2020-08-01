import torch
from torch import nn
import torch.nn.functional as fn


class TorchBasic:
    def __init__(self, function, x):
        self.function = function
        self.X = x

    def GetDervitave(self):
        result = self.function
        result.backward(self.X)
        print(self.X.grad)

    def Scalar(self):
        print(f'torch.tensor(1.)==>{torch.tensor(1.)}')
        print(
            f'type(self.X)==>{type(self.X)}\nself.X.shape==>{self.X.shape}\nself.X.size()==>{self.X.size()}\nlen(self.X)==>{len(self.X)}')
        print(f'self.X.dim()==>{self.X.dim()}')


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = fn.max_pool2d(fn.relu(self.conv1(x)), (2, 2))
        x = fn.max_pool2d(fn.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = fn.relu(self.fc1(x))
        x = fn.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_feature = 1
        for s in size:
            num_feature *= s
        return num_feature


cnn = CNN()
print(cnn)

params = list(cnn.parameters())
# print(len(params))

input = torch.randn(1, 1, 32, 32)
output = cnn(input)
print(output)

# cnn.zero_grad()
# output.backward(torch.randn(1,10))

target=torch.randn(10)
target=target.view(1,-1)
judge=nn.MSELoss()
loss=judge(output,target)
print(loss)





# x = torch.ones(2,3, requires_grad=True)
# function= (x+2)*(x+2)+5
# basic=TorchBasic(function,x)
# basic.GetDervitave()
# basic.Scalar()
