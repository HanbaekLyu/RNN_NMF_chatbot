
import os
import numpy as np
datapath=r'C:\Users\wxwyl\Desktop\wylcode\nmf-train-save\shakes-nmf.npz' #一个文件夹下多个npy文件，
txtpath=r'C:\Users\wxwyl\Desktop\wylcode\nmf-train-save\shakes-nmf.txt'

data = np.load(datapath).reshape([-1, 2])  # (39, 2)
np.savetxt(txtpath,data)

print ('over')
