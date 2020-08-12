
import os
import cv2
import numpy as np
from tqdm import tqdm

REBUILD_DATA=True

class DogsVsCats():
    ImageSize=50
    CATS=r'D:\DataSets\kagglecatsanddogs_3367a\PetImages\Cat'
    DOGS=r'D:\DataSets\kagglecatsanddogs_3367a\PetImages\Dog'
    # CAT ===>0     DOG ===> 1
    LABELS={CATS:0,DOGS:1}
    training_data=[]
    catcount=0
    dogcount=0


    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for filename in tqdm(os.listdir(label)):
                try:
                    path=os.path.join(label,filename)
                    img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                    img=cv2.resize(img,(self.ImageSize,self.ImageSize))
                    self.training_data.append([np.array(img),np.eye(2)[self.LABELS[label]]])
                    if label==self.CATS:
                        self.catcount+=1
                    elif label==self.DOGS:
                        self.dogcount+=1

                except Exception as e:
                    print(e)

        np.random.shuffle(self.training_data)
        np.save("training_data.npy",self.training_data)
        print(f"Total dog image= {self.dogcount}")
        print(f"Total Cat Image= {self.catcount}")


if REBUILD_DATA:
    dogsvcats=DogsVsCats()
    dogsvcats.make_training_data()