from visualize import *
from dataset_utils import *

def start(voxels, name):
    addRoof(voxels)

    min_sum = -1
    min_ind = -1
    
    for i in range(voxels.shape[0]):

        slice = voxels[i, :, :]

        sum = np.sum(slice)
        if sum < min_sum or min_sum == -1:
            min_sum = sum
            min_ind = i

    plt.imshow(voxels[min_ind, :, :], cmap="plasma_r")
    # plt.show()
    plt.savefig("../plots/" + name + ".png")
    plt.close()
