from __future__ import print_function, division
from loss import *
import scipy as sp
import os
import audtorch

def train(model, device, train_loader, optim, opt):
    # training
    model.train()

    total_loss = 0
    total_loss_hr = 0

    for batch_idx, sample in enumerate(train_loader):

        input = sample['roi']
        # target_rv = sample['rv']
        target_hr = sample['hr']
        # target_rv = target_rv.type(torch.FloatTensor)
        target_hr = target_hr.type(torch.FloatTensor)
        input, target_hr = input.to(device), target_hr.to(device)
        optim.zero_grad()

        if opt.multi == 'only_hr':
            output = model(input)
            loss = pearsonr(output, target_hr)

        loss.backward()
        optim.step()

        # running average
        total_loss += loss.item()  #.item gives you a floating point value instead of a tensor

        pred_hr = output.detach().cpu().numpy()
        target_hr = target_hr.detach().cpu().numpy()

    avg_loss_hr = total_loss / len(train_loader)

    return avg_loss_hr, target_hr, pred_hr


def test(model, device, test_loader, opt):
    model.eval()

    total_loss = 0
    pred_hrs = []
    target_rvs = []
    target_hrs = []
    contributions = np.zeros((1, 92, 560))
    baseline = torch.zeros(1, 92, 560)

    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            input = sample['roi']
            # target_rv = sample['rv']
            target_hr = sample['hr']
            # rv_paths = sample['rv_path']


            input, target_hr = input.to(device), target_hr.to(device)

            if opt.multi == 'only_hr':
                output = model(input)
                loss = pearsonr(output, target_hr)

            total_loss += loss.item()  # .item gives you a floating point value instead of a tensor

            pred_hr = output.detach().cpu().numpy()
            target_hr = target_hr.detach().cpu().numpy()

            pred_hrs.append(pred_hr.squeeze())
            target_hrs.append(target_hr.squeeze())

            # pair test sample and its correlation
            # if opt.mode == 'test':
            #     name = rv_paths[0].split('/')[-1].rsplit('.mat')[0]
            #     np.savetx
            #     t('/home/bayrakrg/Data/RV/model_prediction/{}/pred/{}.txt'.format(opt.test_fold.rstrip('.txt'), name), pred.squeeze())
            #     np.savetxt('/home/bayrakrg/Data/RV/model_prediction/{}/gt/{}.txt'.format(opt.test_fold.rstrip('.txt'), name), target_rv.squeeze())
            #
            #     corr_coeff = sp.stats.pearsonr(pred.squeeze(), target_rv.squeeze())
            #     file.write(str(corr_coeff[0]))
            #     file.write('\n')

        avg_loss = total_loss / len(test_loader)

    return avg_loss, target_hrs, pred_hrs