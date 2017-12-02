import tensorflow as tf
import numpy as np
import PIL.Image

N = 100

u_init = np.zeros([N, N], dtype=np.float32)
ut_init = np.zeros([N, N], dtype=np.float32)

im = PIL.Image.open('test100.jpg').convert('L')
(width, height) = im.size
gm = list(im.getdata())
gm = np.array(gm)
gm = gm.reshape((height, width))

for n in range(100):
  a,b = np.random.randint(0, N, 2)
  u_init[a,b] = np.random.uniform()

x=u_init+gm


img_out = PIL.Image.fromarray((x * 255).astype(np.uint8))
img_out.show()
