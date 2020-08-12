# import cv2
# import numpy as np
#
# path=r'C:\PycharmProjects\pytorch\cnn_food_classification\food-11\testing\0000.jpg'
# img=cv2.imread(path)
# imgclean=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# imgclean=cv2.resize(imgclean,(38,51))
# cv2.imwrite('new.jpg',imgclean)


import matplotlib.pyplot as plt
plt.imread('new.jpg')
plt.show()

# with open('imgggg','w') as f:
#     for i in imgclean:
#         for j in i:
#             f.write(str(j)+" ")
#         f.write('\n')
