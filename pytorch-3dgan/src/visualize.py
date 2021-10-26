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

def getVoxelFromMat(path, cube_len=64):
    if cube_len == 32:
        voxels = io.loadmat(path)['voxel'] # 30x30x30
        voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))

    else:
        # voxels = np.load(path) 
        # voxels = io.loadmat(path)['instance'] # 64x64x64
        # voxels = np.pad(voxels, (2, 2), 'constant', constant_values=(0, 0))
        # print (voxels.shape)
        voxels = io.loadmat(path)['voxel'] # 30x30x30
        print(voxels.shape)
        print(voxels.__class__)
        voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
        voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
        # print ('here')
    # print (voxels.shape)
    return voxels


def saveMatfromVoxel(path, voxels):
    mdict = {"a": voxels}
    io.savemat(path, mdict)

def saveSlicesfromVoxel(path, voxels, axis):

    for i in range(38, 45):#, voxels.shape[axis]):

        if axis == 0:
            slice = voxels[i, :, :]
        elif axis == 1:
            slice = voxels[:, i, :]
        elif axis == 2:
            slice = voxels[:, :, i]

        plt.imshow(slice, cmap="plasma_r")
        plt.show()
        # plt.savefig(path + 'test123.png')
        plt.close()


def getVFByMarchingCubes(voxels, threshold=0.8):
    v, f = sk.marching_cubes_classic(voxels, level=threshold)
    return v, f

#voxels je ndarray 32x32x32 (ili slicno)
def plotVoxelVisdom(voxels, visdom, title):
    v, f = getVFByMarchingCubes(voxels)
    visdom.mesh(X=v, Y=f, opts=dict(opacity=0.5, title=title))


def SavePloat_Voxels(voxels, path, iteration):
    voxels = voxels[:8].__ge__(0.5)
    fig = plt.figure(figsize=(32, 16))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(voxels):
        x, y, z = sample.nonzero()
        ax = plt.subplot(gs[i], projection='3d')
        ax.scatter(x, y, z, zdir='z', c='red')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.set_aspect('equal')
    # print (path + '/{}.png'.format(str(iteration).zfill(3)))
    plt.savefig(path + '000test.png')
    # plt.savefig(path + '/{}.png'.format(str(iteration).zfill(3)), bbox_inches='tight')
    plt.close()


def tester(args):

    image_saved_path = '../images'
    # image_saved_path = params.images_dir
    if not os.path.exists(image_saved_path):
        os.makedirs(image_saved_path)

    vis = visdom.Visdom()


def main():

    path = sys.argv[1]

    print("path: ", path)

    vis = visdom.Visdom()

    
    voxels = io.loadmat(path)['a']

    # print(voxels[1, 0:21, 13])

    saveSlicesfromVoxel("blip", voxels, 0)

    plotVoxelVisdom(voxels, vis, path)

    """
    count = 10

    for filename in os.listdir(path):
        if filename.endswith(".mat"):
            if count < 0:
                break
            # elif count % 100 != 0:
            #     continue
            print(filename) 
            voxels = io.loadmat(path + '/' + filename)['a']
            # SavePloat_Voxels(np.expand_dims(voxels, axis=0), ".", 0)
            plotVoxelVisdom(voxels, vis, path)
            count -= 1
        else:
            continue
    """



if __name__ == '__main__':
    main()
