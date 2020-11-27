from __future__ import print_function, division
from loss import *
import scipy as sp
import os
import audtorch

def test(model, device, test_loader, opt):
    model.eval()

    total_loss = 0
    pred_rvs = []
    pred_hrs = []
    target_rvs = []
    target_hrs = []

    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            input = sample['roi']
            target_rv = sample['rv']
            target_hr = sample['hr']
            hr_paths = sample['hr_path']

            input, target_rv, target_hr = input.to(device), target_rv.to(device), target_hr.to(device)

            output_rv, output_hr = model(input)
            loss_rv = pearsonr(output_rv, target_rv)
            loss_hr = pearsonr(output_hr, target_hr)
            loss = opt.l1 * loss_rv + opt.l2 * loss_hr

            # running average
            # total_loss_rv += loss_rv.item()
            # total_loss_hr += loss_hr.item()
            total_loss += loss.item()  # .item gives you a floating point value instead of a tensor

            # convert torch to numpy array
            pred_rv = output_rv.detach().cpu().numpy()
            pred_hr = output_hr.detach().cpu().numpy()
            # print(pred.shape)
            target_rv = target_rv.detach().cpu().numpy()
            target_hr = target_hr.detach().cpu().numpy()

            pred_rvs.append(pred_rv.squeeze())
            pred_hrs.append(pred_hr.squeeze())
            target_rvs.append(target_rv.squeeze())
            target_hrs.append(target_hr.squeeze())

            # if opt.mode == 'test':
            #     name = rv_paths[0].split('/')[-1].rsplit('.mat')[0]
            #     np.savetxt('/home/bayrakrg/Data/RV/model_prediction/{}/pred/{}.txt'.format(opt.test_fold.rstrip('.txt'), name), pred.squeeze())
            #     np.savetxt('/home/bayrakrg/Data/RV/model_prediction/{}/gt/{}.txt'.format(opt.test_fold.rstrip('.txt'), name), target_rv.squeeze())
            #
            #     corr_coeff = sp.stats.pearsonr(pred.squeeze(), target_rv.squeeze())
            #     file.write(str(corr_coeff[0]))
            #     file.write('\n')

        avg_loss = total_loss / len(test_loader)

    return avg_loss, target_rvs, target_hrs, pred_rvs, pred_hrs