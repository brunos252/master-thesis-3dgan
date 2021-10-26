import os
import sys
import matplotlib.pyplot as plt

def main():

    path = sys.argv[1]

    print("path: ", path)    

    file1 = open(path, 'r')
    
    Lines = file1.readlines()
    x = ([])
    y1 = ([])
    y2 = ([])

    
    for line in Lines:
        if line.startswith("Epochs"):
            seg = line.strip().split(" ")
            x.append(float(seg[0][7:]))
            y1.append(float(seg[5][:-1]))
            y2.append(float(seg[8]))
            print("{} : {} {}".format(x, y1, y2))

    plt.scatter(x, y1)
    plt.plot(x, y1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    file1.close()

if __name__ == '__main__':
    main()