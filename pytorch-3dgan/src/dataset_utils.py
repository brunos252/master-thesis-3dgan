from numpy import add
from visualize import *
from tqdm import tqdm
import numpy as np
import sys, os

def removeRoof(voxels):
    voxels[:, 0:21, :] = 255

def addRoof(voxels):
    voxels[:, 0:19, :] = 0.608
    voxels[:, 19, :] = 0.737
    voxels[:, 20, :] = 0.871

def normalizeVoxels(voxels):
    return np.float32(voxels / 255)

# argumenti: <action> <path>
# action: {add_roof, remove_roof, scale}
# path: dir koji sadrzi datoteke
def main():
    
    action = sys.argv[1]
    path = sys.argv[2]
    if action == "add_roof":
        output_path = path + "_roofed"
    elif action == "remove_roof":
        output_path = path + "_roofless"
    elif action == "scale":
        output_path = path + "_scaled"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print("input path: ", path)
    print("output path: ", output_path)

    files = [f for f in os.listdir(path) if f.endswith(".mat")]

    for filename in tqdm(files):
            voxels = io.loadmat(path + '/' + filename)['a']

            if action == "add_roof":
                addRoof(voxels)
            elif action == "remove_roof":
                removeRoof(voxels)
            elif action == "scale":
                voxels = normalizeVoxels(voxels)

            saveMatfromVoxel(output_path + '/' + filename, voxels)

if __name__ == '__main__':
    main()