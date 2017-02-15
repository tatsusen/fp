# -*- coding: utf-8 -*-
import sys, os
import numpy as np
import mnist
from PIL import Image

print('-----------------------')
print(os.path.dirname(__file__))

a1=((10,20),(30,40),(50,60))
a2=((1,2),(3,4))
a3=np.array(a2)
print(a1)
print(a2)
print(a3)
print(a2*a3)
print(np.dot(a1,a2))

(x_tr, t_tr),(x_test, t_test) =mnist.load_mnist(normalize=False,flatten=True)

img=x_tr[9]
label=t_tr[9]
print(label)
print(img.shape)
img=img.reshape(28,28)
print(img.shape)
pil_img=Image.fromarray(np.uint8(img))
pil_img.show()
