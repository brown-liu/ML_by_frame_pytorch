import torch
import time
a=time.time()
li1=[]

li2=[]

for i in range(10000):
    x=torch.tensor(i)
    result1=(x**2)


b=time.time()

for i in range(10000):
    y = torch.tensor(i)
    result2=(y ** 2).cuda()


c=time.time()

print(f"first run {b-a}sconds and second run{c-b} seconds")