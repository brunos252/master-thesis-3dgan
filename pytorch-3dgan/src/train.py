
import torch
from torch import optim
from torch import  nn
from utils import *
import os

from model import net_G, net_D

import time
import params
from tqdm import tqdm

def train(args):

    save_file_path = params.output_dir + '/' + args.model_name
    print (save_file_path)  # ../outputs/dcgan
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)
        
    if not os.path.exists(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
   
    if not os.path.exists(params.images_dir):
        os.makedirs(params.images_dir)

    dsets_path = params.data_dir + params.model_dir # + "30/train/"

    print (dsets_path)

    train_dsets = ShapeNetDataset(dsets_path, args, "train")
    
    train_dset_loaders = torch.utils.data.DataLoader(train_dsets, batch_size=params.batch_size, shuffle=True, num_workers=1)
    
    dset_len = {"train": len(train_dsets)}
    dset_loaders = {"train": train_dset_loaders}

    # model define
    D = net_D(args)
    G = net_G(args)

    D_solver = optim.Adam(D.parameters(), lr=params.d_lr, betas=params.beta)
    G_solver = optim.Adam(G.parameters(), lr=params.g_lr, betas=params.beta)

    criterion_D = nn.BCELoss()
    criterion_G = nn.L1Loss()

    #load checkpoint =====================================================================
    start_epoch = 0

    if args.use_checkpoint == True:

        for filename in os.listdir(params.checkpoint_dir):
            if filename.startswith("checkpoint_"):
                check_epoch = int(filename[11:-4])
                print("checkpoint epoch: ", check_epoch)
                if check_epoch > start_epoch:
                    start_epoch = check_epoch
                    
        if start_epoch != 0:

            save_file_path = params.output_dir + '/' + args.model_name
            pretrained_file_path_G = save_file_path + '/' + 'G_' + str(start_epoch) + '.pth'
            pretrained_file_path_D = save_file_path + '/' + 'D_' + str(start_epoch) + '.pth'

            if not torch.cuda.is_available():
                G.load_state_dict(torch.load(pretrained_file_path_G, map_location={'cuda:0': 'cpu'}))
                D.load_state_dict(torch.load(pretrained_file_path_D, map_location={'cuda:0': 'cpu'}))
                checkpoint = torch.load(params.checkpoint_dir + '/checkpoint_' + str(start_epoch) + '.tar', map_location={'cuda:0': 'cpu'})
            else:
              G.load_state_dict(torch.load(pretrained_file_path_G))
              D.load_state_dict(torch.load(pretrained_file_path_D))
              D.to(params.device)
              G.to(params.device)
              criterion_D.to(params.device)
              criterion_G.to(params.device)
              print("trazimo: ", params.checkpoint_dir + '/checkpoint_' + str(start_epoch) + '.tar')
              checkpoint = torch.load(params.checkpoint_dir + '/checkpoint_' + str(start_epoch) + '.tar')

            D_solver.load_state_dict(checkpoint['D_optimizer_state_dict'])
            G_solver.load_state_dict(checkpoint['G_optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            criterion_D = checkpoint['loss_D']
            criterion_G = checkpoint['loss_G']

    else:
        D.to(params.device)
        G.to(params.device)

    # ==================================================================================

    itr_val = -1
    itr_train = -1

    for epoch in range(start_epoch, params.epochs):

        start = time.time()
        
        for phase in ['train']:
            if phase == 'train':
                D.train()
                G.train()
            else:
                D.eval()
                G.eval()

            running_loss_G = 0.0
            running_loss_D = 0.0
            running_loss_adv_G = 0.0

            for i, X in enumerate(tqdm(dset_loaders[phase])):

                if phase == 'train':
                    itr_train += 1

                X = X.to(params.device)
                
                batch = X.size()[0]

                Z = generateZ(args, batch)

                # Train discriminator ===========================
                d_real = D(X)

                fake = G(Z)
                d_fake = D(fake)

                real_labels = torch.ones_like(d_real).to(params.device)
                fake_labels = torch.zeros_like(d_fake).to(params.device)
                # print (d_fake.size(), fake_labels.size())

                if params.soft_label:
                    real_labels = torch.Tensor(batch).uniform_(0.7, 1.2).to(params.device)
                    fake_labels = torch.Tensor(batch).uniform_(0, 0.3).to(params.device)

                # print (d_real.size(), real_labels.size())
                d_real_loss = criterion_D(d_real, real_labels)
                

                d_fake_loss = criterion_D(d_fake, fake_labels)

                d_loss = d_real_loss + d_fake_loss

                d_real_acu = torch.ge(d_real.squeeze(), 0.5).float()
                d_fake_acu = torch.le(d_fake.squeeze(), 0.5).float()
                d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu),0))

                if d_total_acu < params.d_thresh:
                    D.zero_grad()
                    d_loss.backward()
                    D_solver.step()

                # Train generator ==============================
                
                Z = generateZ(args, batch)

                # print (X)
                fake = G(Z)
                d_fake = D(fake)

                adv_g_loss = criterion_D(d_fake, real_labels)

                recon_g_loss = criterion_G(fake, X)
                g_loss = adv_g_loss

                if args.local_test:
                    print('Iteration-{} , D(x) : {:.4}, D(G(x)) : {:.4}'.format(itr_train, d_loss.item(), adv_g_loss.item()))

                D.zero_grad()
                G.zero_grad()
                g_loss.backward()
                G_solver.step()


                running_loss_G += recon_g_loss.item() * X.size(0)
                running_loss_D += d_loss.item() * X.size(0)
                running_loss_adv_G += adv_g_loss.item() * X.size(0)
           

            epoch_loss_G = running_loss_G / dset_len[phase]
            epoch_loss_D = running_loss_D / dset_len[phase]
            epoch_loss_adv_G = running_loss_adv_G / dset_len[phase]


            end = time.time()
            epoch_time = end - start


            print('Epochs-{} ({}) , D(x) : {:.4}, D(G(x)) : {:.4}'.format(epoch, phase, epoch_loss_D, epoch_loss_adv_G))
            print ('Elapsed Time: {:.4} min'.format(epoch_time/60.0))

            if (epoch + 1) % params.model_save_step == 0:

                # print ('model_saved, images_saved...')
                torch.save(G.state_dict(), params.output_dir + '/' + args.model_name + '/' + 'G_' + str(epoch) + '.pth')
                torch.save(D.state_dict(), params.output_dir + '/' + args.model_name + '/' + 'D_' + str(epoch) + '.pth')

                # checkpoint save =====================================================================
                
                torch.save({
                    'epoch': epoch,
                    'G_optimizer_state_dict': G_solver.state_dict(),
                    'D_optimizer_state_dict': D_solver.state_dict(),
                    'loss_G': criterion_G,
                    'loss_D': criterion_D
                }, params.checkpoint_dir + '/checkpoint_' + str(epoch) + '.tar')

                print('model_saved, images_saved, checkpoint saved...')

                # ==================================================================================

                samples = fake.cpu().data[:8].squeeze().numpy()

                SavePloat_Voxels(samples, params.images_dir, epoch)



                






