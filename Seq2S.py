import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
import math
import pandas as pd


# Backbone conv net.
# Zhang, Kai, et al. "Beyond a gaussian denoiser: Residual learning of deep cnn for
# image denoising." IEEE transactions on image processing 26.7 (2017): 3142-3155.
# https://github.com/SaoYan/DnCNN-PyTorch/blob/master/models.py,
class DnCNN(nn.Module):
    def __init__(self, channels=1, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        return out


# Sequence image set; dataset_path takes numpy files shaped (N, H, W) in [0,1], in sequential order.
class Sequential_Dataset(Dataset):
    def __init__(self, dataset_path, p=0.7):
        df = torch.tensor(np.load(dataset_path)).float()/255
        self.df1 = df[:-2,:,:]
        self.df2 = df[1:-1,:,:]
        self.df3 = df[2:,:,:]
        self.p = p

    def __len__(self):
        return self.df1.size(0)

    def __getitem__(self, index):
        x1 = self.df1[index, :, :].unsqueeze(0)
        x2 = self.df2[index, :, :].unsqueeze(0)
        x3 = self.df3[index, :, :].unsqueeze(0)
        bern_map = torch.bernoulli(torch.full_like(x2, self.p))
        x2_drop = torch.cat((x1, x2 * bern_map, x3), dim=0)
        return {'input': x2, 'stacked input': x2_drop, 'dropped mask': 1-bern_map}


# Train & inference
def Seq2S(train_set_path, test_set_path, save_path,
          p=0.3, batch_size=1, max_epoch=25,
          adam_lr=1e-4, adam_beta1=0.5, lr_sched_ss=1, lr_sched_gamma=0.9,
          early_stop_patience=3, stop_threshold=1e-4,):
    assert(torch.cuda.is_available()), 'No gpu or cuda not enabled.'

    t_very_start=time.time()
    device = 'cuda:0'
    os.makedirs(save_path, exist_ok=True)

    # Model & optimizer
    model = DnCNN(channels=3)
    model.to(device)
    if torch.cuda.device_count() >= 1:
        model = nn.DataParallel(model)
        model.to(device)

    opt = optim.Adam(model.parameters(), lr=adam_lr, betas=(adam_beta1, 0.99))
    sched = optim.lr_scheduler.StepLR(opt, step_size=lr_sched_ss, gamma=lr_sched_gamma)

    # Datasets:
    trSet = Sequential_Dataset(train_set_path, p=1. - p)
    teSet = Sequential_Dataset(test_set_path, p=1. - p)

    trLoader = DataLoader(trSet, batch_size=batch_size, shuffle=True)
    teLoader = DataLoader(teSet, batch_size=batch_size, shuffle=False)

    # Start training:
    t = trange(max_epoch)
    t_start = time.time()
    print('Training:')

    trobj_list, teobj_list, iteration_list = [], [], []
    iteration, patience, teobj_min = 0, 0, 1e10

    for e in t:

        # early stop, if applicable.
        if not math.isnan(early_stop_patience):
            model.eval()
            teobj_avg = 0
            for batch in teLoader:
                seq = batch['stacked input']
                s = batch['input']
                bm = batch['dropped mask']

                teobj_avg += F.mse_loss(bm * model(seq.to(device)).detach().cpu(),
                                        bm * s
                                        ).item() * seq.size(0) / len(teSet)
            if teobj_min - teobj_avg > stop_threshold:
                patience = 0
                teobj_min = teobj_avg
            else:
                patience += 1

            if patience > early_stop_patience:
                break
        else:
            patience, teobj_min, teobj_avg = float('nan'), float('nan'), float('nan')

        model.train()
        trobj_avg = 0.
        for batch in trLoader:
            opt.zero_grad()

            seq = batch['stacked input'].to(device)
            s = batch['input'].to(device)
            bm = batch['dropped mask'].to(device)

            denoised = model(seq)
            obj = F.mse_loss(bm*denoised, bm*s)
            obj.backward()
            opt.step()

            iteration += 1
            one_iteration_time = (time.time()-t_start)/iteration
            ETAm = ((len(trLoader)*max_epoch - iteration)*one_iteration_time)/60

            trobj_avg += obj.item()*seq.size(0)/len(trSet)

            t.set_description(
                f'#{e}/{max_epoch} ({patience}/{early_stop_patience}), '
                f'Time:{(time.time()-t_start)/60: .2f} m (ETA:{ETAm: .2f} m), '
                f'Loss: Train batch={obj.item(): .4f}, Test set={teobj_avg: .4f} (>{teobj_min: .4f})', refresh=True)

        sched.step()

        iteration_list += [iteration]
        trobj_list += [trobj_avg]
        teobj_list += [teobj_avg]

    torch.save(model.state_dict(), save_path + f"/Model W&B")

    # Fianl inference.
    print("Inference:")

    inf_loader = DataLoader(teSet, batch_size=1, shuffle=False)
    inf_save_path = save_path + f"/Denoised images"
    os.makedirs(inf_save_path, exist_ok=True)

    model.eval()
    inf_out_list = []
    topil = transforms.ToPILImage()

    for _ in tqdm(range(10)):
        mcn_out_batch = []
        for b_i, batch in enumerate(inf_loader, 0):
            seq = batch['stacked input'].to(device)
            mcn_out_batch += [model(seq).detach().cpu()]
        inf_out_list += [torch.cat(mcn_out_batch, dim=0)]

    final_result = torch.stack(inf_out_list, 0).mean(0)

    for i in range(final_result.size(0)):
        denoised = topil(final_result[i,:])
        denoised.save(inf_save_path+f"/{i}.png")

    tot_time = (time.time()-t_very_start)
    print(f"Finished, ({tot_time/60: .2f} m)")

    pd.DataFrame({'iteration': iteration_list, 'trOBJ': trobj_list, 'teOBJ': teobj_list}
                 ).to_csv(save_path+f"/Log_{int(tot_time)}s.csv", index=False)



if __name__ == '__main__':

    fuel_train_set_path = f'F24_dataset_test.npy'
    fuel_test_set_path = f'F24_dataset_test.npy'

    Seq2S(train_set_path=fuel_train_set_path,
          test_set_path=fuel_test_set_path,
          save_path=f'F24_Seq2S_30%',
          p=0.3)