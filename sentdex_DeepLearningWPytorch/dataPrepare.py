import torch
import os
import torchvision
from torchvision import transforms,datasets
from sentdex_DeepLearningWPytorch.NNcore import nnModule
import torch.nn.functional as F

train=datasets.MNIST(r"D:\dataset",train=True,download=True,transform=transforms.Compose([transforms.ToTensor()]))
test=datasets.MNIST(r"D:\dataset",train=False,download=True,transform=transforms.Compose([transforms.ToTensor()]))

trainset=torch.utils.data.DataLoader(train,batch_size=10,shuffle=True)
testset=torch.utils.data.DataLoader(test,batch_size=10,shuffle=True)

for data in trainset:
    pass


x,y=data[0][0], data[1][0]

# print(x.shape)
#
# import matplotlib.pyplot as plt
# plt.imshow(data[0][0].view(28,28))
# plt.show()
#
# total=0
# counter_dict={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
# for data in trainset:
#     Xs,ys=data
#     for y in ys:
#         total+=1
#         counter_dict[int(y)] +=1





# print(net)
#
# X=torch.randn((28,28))
# X=X.view(1,28*28)
#
# out=net.forward(X)
# print(out)

net=nnModule.Net()
import torch.optim as optim
optimizer=optim.Adam(net.parameters(),lr=0.001)


EPOCH=3
for epoch in range(EPOCH):
    for data in trainset:
        X,y=data
        net.zero_grad()
        output=net(X.view(-1,28*28))
        loss=F.nll_loss(output,y)
        loss.backward()
        optimizer.step()
    print(loss)


correct=0
total=0
with torch.no_grad():
    for data in trainset:
        X,y=data
        output=net(X.view(-1,784))
        for idx,i in enumerate(output):
            if torch.argmax(i)==y[idx]:
                correct+=1
            total+=1

print(f"Accuracy: {round(correct/total,3)}")