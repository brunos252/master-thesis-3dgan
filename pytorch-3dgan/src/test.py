import torch
from utils import *
import os
from model import net_G, net_D

import params
import visdom
import visualize_2D
import time, math

def test(args):
    print ('Evaluation Mode...')

    image_saved_path = '../images'
    if not os.path.exists(image_saved_path):
        os.makedirs(image_saved_path)

    if args.use_visdom == True:
        vis = visdom.Visdom()


    save_file_path = params.output_dir + '/' + args.model_name

    latest_epoch = 0

    for filename in os.listdir(save_file_path):
        if filename.startswith("D_"):
            check_epoch = int(filename[2:-4])
            if check_epoch > latest_epoch:
                latest_epoch = check_epoch

    print("using model from epoch ", latest_epoch)

    pretrained_file_path_G = save_file_path+'/'+'G_' + str(latest_epoch) + '.pth'
    pretrained_file_path_D = save_file_path+'/'+'D_' + str(latest_epoch) + '.pth'

    
    D = net_D(args)
    G = net_G(args)


    if not torch.cuda.is_available():
        G.load_state_dict(torch.load(pretrained_file_path_G, map_location={'cuda:0': 'cpu'}))
        D.load_state_dict(torch.load(pretrained_file_path_D, map_location={'cuda:0': 'cpu'}))
    else:
        G.load_state_dict(torch.load(pretrained_file_path_G))
        D.load_state_dict(torch.load(pretrained_file_path_D, map_location={'cuda:0': 'cpu'}))
    
    print ('visualizing model')
    
    G.to(params.device)
    D.to(params.device)
    G.eval()
    D.eval()

    N = 8

    tmstmp = math.floor(time.time()) % 10000

    for i in range(N):
        
        z = generateZ(args, 1)
        
        fake = G(z)
        
        samples = fake.unsqueeze(dim=0).detach().numpy()

        # visualization
        if args.use_visdom == False:
            SavePloat_Voxels(samples, image_saved_path, 'tester_norm_'+str(i))
        else:
            # mdict = {"a": samples[0, :]}
            # io.savemat("generated_" + str(i) + ".mat", mdict)
            plotVoxelVisdom(samples[0,:], vis, "tester_"+str(i))

        if args.make_2D == True:
            visualize_2D.start(samples[0, :], "epoch_" + str(latest_epoch) + "_" + str(tmstmp) + "_example_" + str(i))
        
        


