from __future__ import print_function, division
from loss import *
import scipy as sp
import os
import audtorch


def train(model, device, train_loader, optim, opt):
    # training
    model.train()

    total_loss = 0
    total_loss_rv = 0
    total_loss_hr = 0
    temp_att = []
    spat_att = []

    for batch_idx, sample in enumerate(train_loader):

        input = sample['roi']
        target_rv = sample['rv']
        target_hr = sample['hr']
        target_rv = target_rv.type(torch.FloatTensor)
        target_hr = target_hr.type(torch.FloatTensor)
        input, target_rv, target_hr = input.to(device), target_rv.to(device), target_hr.to(device)
        optim.zero_grad()
        output_rv, output_hr, t_att, s_att = model(input)

        loss_rv = pearsonr(output_rv, target_rv)
        loss = opt.l1 * loss_rv
        total_loss_rv += loss_rv.item()

        loss_hr = pearsonr(output_hr, target_hr)
        loss += opt.l2 * loss_hr
        total_loss_hr += loss_hr.item()

        loss.backward()
        optim.step()

        # running average
        total_loss += loss.item()  #.item gives you a floating point value instead of a tensor


    # convert torch to numpy array
    pred_rv = output_rv.detach().cpu().numpy()
    pred_hr = output_hr.detach().cpu().numpy()

    target_rv = target_rv.detach().cpu().numpy()
    target_hr = target_hr.detach().cpu().numpy()

    avg_loss_rv = total_loss_rv / len(train_loader)
    avg_loss_hr = total_loss_hr / len(train_loader)

    t_att = t_att.detach().cpu().numpy()
    s_att = s_att.detach().cpu().numpy()
    temp_att.append(t_att.squeeze())
    spat_att.append(s_att.squeeze())

    avg_loss = total_loss / len(train_loader)

    return avg_loss, avg_loss_rv, avg_loss_hr, target_rv, target_hr, pred_rv, pred_hr, temp_att, spat_att


def test(model, device, test_loader, opt):
    model.eval()

    total_loss = 0
    pred_rvs = []
    pred_hrs = []
    target_rvs = []
    target_hrs = []
    temp_att = []
    spat_att = []

    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            input = sample['roi']
            target_rv = sample['rv']
            target_hr = sample['hr']
            # schaefer_paths = sample['schaefer']

            input, target_rv, target_hr = input.to(device), target_rv.to(device), target_hr.to(device)

            # output_rv, output_hr = model(input)
            output_rv, output_hr, t_att, s_att = model(input)
            loss_rv = pearsonr(output_rv, target_rv)
            loss_hr = pearsonr(output_hr, target_hr)
            loss = opt.l1 * loss_rv + opt.l2 * loss_hr

            # running average
            total_loss += loss.item()  # .item gives you a floating point value instead of a tensor

            # convert torch to numpy array
            pred_rv = output_rv.detach().cpu().numpy()
            pred_hr = output_hr.detach().cpu().numpy()
            target_rv = target_rv.detach().cpu().numpy()
            target_hr = target_hr.detach().cpu().numpy()

            pred_rvs.append(pred_rv.squeeze())
            pred_hrs.append(pred_hr.squeeze())
            target_rvs.append(target_rv.squeeze())
            target_hrs.append(target_hr.squeeze())

            t_att = t_att.detach().cpu().numpy()
            s_att = s_att.detach().cpu().numpy()
            temp_att.append(t_att.squeeze())
            spat_att.append(s_att.squeeze())

        avg_loss = total_loss / len(test_loader)

    return avg_loss, target_rvs, target_hrs, pred_rvs, pred_hrs, temp_att, spat_att