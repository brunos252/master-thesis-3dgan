import scipy.io as io
import scipy.ndimage as nd
import numpy as np
import skimage.measure as sk
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import visdom
import sys
from visualize import plotVoxelVisdom
from visualize_2D import start

def main():

    path = sys.argv[1]

    print("path: ", path)

    vis = visdom.Visdom()

    for filename in os.listdir(path):
        if filename.endswith(".mat"):
            print(filename) 
            voxels = io.loadmat(path + '/' + filename)['a']
            # SavePloat_Voxels(np.expand_dims(voxels, axis=0), ".", 0)
            plotVoxelVisdom(voxels, vis, path)
            start(voxels, filename)
        else:
            continue
    



if __name__ == '__main__':
    main()
