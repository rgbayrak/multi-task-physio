from __future__ import print_function, division
import torch
from tqdm import tqdm


def train(model, device, train_loader, optim, opt):
    # training
    model.train()

    total_loss = 0
    total_loss_rv = 0
    total_loss_hr = 0
    tloss = []

    for batch_idx, sample in enumerate(train_loader):
        input = sample['roi']
        target_rv = sample['rv']
        target_hr = sample['hr']
        target_rv = target_rv.type(torch.FloatTensor)
        target_hr = target_hr.type(torch.FloatTensor)
        input, target_rv, target_hr = input.to(device), target_rv.to(device), target_hr.to(device)
        optim.zero_grad()
        output_rv, output_hr = model(input)
        criterion = torch.nn.MSELoss()
        # print(target.shape)
        # print(output.shape)

        loss_rv = criterion(output_rv, target_rv)
        loss_hr = criterion(output_hr, target_hr)
        loss = opt.l1 * loss_rv + (1-opt.l1) * loss_hr
        loss.backward()
        optim.step()

        # running average
        total_loss_rv += loss_rv.item()
        total_loss_hr += loss_hr.item()
        total_loss += loss.item()  #.item gives you a floating point value instead of a tensor

    avg_loss = total_loss / len(train_loader)
    tloss.append(avg_loss)
    return avg_loss, tloss


def test(model, device, test_loader, opt):
    model.eval()

    total_loss = 0
    pred_rvs = []
    target_rvs = []
    pred_hrs = []
    target_hrs = []

    with torch.no_grad():
        # with tqdm(total=len(test_loader)) as pbar:
        for batch_idx, sample in enumerate(test_loader):
            input = sample['roi']
            target_rv = sample['rv']
            target_hr = sample['hr']

            # CHANGE THIS DURING TESTING
            # if sliding window
            if opt.mode == 'test':
                input = input.squeeze()
                target_rv = target_rv.squeeze()
                target_hr = target_hr.squeeze()

            target_rv = target_rv.type(torch.FloatTensor)
            target_hr = target_hr.type(torch.FloatTensor)
            input, target_rv, target_hr = input.to(device), target_rv.to(device), target_hr.to(device)
            output_rv, output_hr = model(input)
            # print(target.shape)
            # print(output.shape)

            if opt.mode == 'test':
                output_rv = output_rv.squeeze()
                output_hr = output_hr.squeeze()
            # target = target.squeeze()

            criterion = torch.nn.MSELoss()
            loss_rv = criterion(output_rv, target_rv)
            loss_hr = criterion(output_hr, target_hr)
            loss = opt.l1 * loss_rv + (1 - opt.l1) * loss_hr
            # print(loss)

            # convert torch to numpy array and save predictions
            pred_rvs.append(output_rv.cpu().numpy())
            pred_hrs.append(output_hr.cpu().numpy())
            target_rvs.append(target_rv.cpu().numpy())
            target_hrs.append(target_hr.cpu().numpy())


        total_loss += loss.item()

        avg_loss = total_loss / len(test_loader)

    return avg_loss, target_rvs, pred_rvs, target_hrs, pred_hrs