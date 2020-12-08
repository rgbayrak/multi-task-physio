import torch
import torchvision.transforms as transforms
import argparse
from torch.utils.data import DataLoader
# from data_loader import *
from data_loader_select import *
from trainer import *
from model import *
from tqdm import tqdm
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
import csv
import os



# def print_network(net):
#     num_params = 0
#     for param in net.parameters():
#         num_params += param.numel()
#     print(net)
#     print('Total number of parameters: %d' % num_params)


def train_model(opt):
    # device CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = {'num_workers': 2, 'pin_memory': True} if torch.cuda.is_available() else {}

    # create fold specific dictionaries for train and validation split
    train_data = get_dictionary(opt.train_fold)
    keys = list(train_data.keys())

    # calculate number of regs which becomes the channel
    chs = get_roi_len(opt.roi_list)

    # assign random validation remove them from train data
    val_split = round(len(train_data) * opt.val_split)
    val_data = {}
    for i in range(val_split):
        idx = random.randint(0, len(keys) - 1)
        val_data[keys[idx]] = train_data[keys[idx]]
        del train_data[keys[idx]]
        del keys[idx]

    # load the train/val data as tensor
    train_set = data_to_tensor(train_data, opt.roi_list, opt.percent)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=opt.train_batch, shuffle=True, **kwargs)

    val_set = data_to_tensor(val_data, opt.roi_list, opt.percent)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=1, shuffle=True, **kwargs)

    # load network
    if opt.model == 'Bi-LSTM':
        model = BidirectionalLSTM(chs, 2048, 1)
    else:
        print('Error!')


    optim = torch.optim.Adam(model.parameters(), lr=opt.lr)
    train_loss_file = '{}results/{}/train_loss_split_{}'.format(opt.out_dir, opt.uni_id, opt.train_fold)
    f = open(train_loss_file, 'w')
    f.close()
    validate_loss_file = '{}results/{}/validate_loss_split_{}'.format(opt.out_dir, opt.uni_id, opt.train_fold)
    f = open(validate_loss_file, 'w')
    f.close()

    model_file = '{}models/{}/saved_model_split_{}'.format(opt.out_dir, opt.uni_id, opt.train_fold)

    seq_increase = 0
    min_loss = 10000

    if opt.continue_training:
        model.load_state_dict(torch.load(model_file))
        model = model.to(device)
    else:
        model = model.to(device)

    with tqdm(total=opt.epoch) as pbar:
        for epoch in range(1, opt.epoch + 1):

            avg_loss, avg_loss_rv, avg_loss_hr, target_rv, target_hr, pred_rv, pred_hr = train(model, device, train_loader, optim, opt)

            avg_val_loss, target_rvs, target_hrs, pred_rvs, pred_hrs = test(model, device, val_loader, opt)

            # # plot prediction vs output
            # plt.figure(figsize=(15.5, 5))
            #
            # n = 0
            # target = target_hr[n][:]
            # hr = pred_hr[n][:]
            # thr = (target - target.mean(axis=0)) / target.std(axis=0)  # z-score normalization
            # phr = (hr - hr.mean(axis=0)) / hr.std(axis=0)  # z-score normalization
            #
            # target = target_rv[n][:]
            # hr = pred_rv[n][:]
            # trv = (target - target.mean(axis=0)) / target.std(axis=0)  # z-score normalization
            # prv = (hr - hr.mean(axis=0)) / hr.std(axis=0)  # z-score normalization
            #
            # plt.subplot(211)
            # plt.plot(np.arange(0, 560), phr)
            # plt.plot(np.arange(0, 560), thr)
            # plt.ylabel('hr')
            # plt.legend(['Prediction', 'Target'])
            # plt.subplot(212)
            # plt.plot(np.arange(0, 560), prv)
            # plt.plot(np.arange(0, 560), trv)
            # plt.ylabel('rv')
            # plt.legend(['Prediction', 'Target'])
            # plt.show()

            with open(train_loss_file, "a") as file:
                file.write(str(avg_loss_hr))
                file.write('\n')

            with open(validate_loss_file, "a") as file:
                file.write(str(avg_val_loss))
                file.write('\n')

            # save model only if validation loss is lower than prev. saved model
            if avg_val_loss < min_loss:
                min_loss = avg_val_loss
                with open(model_file, 'wb') as f:
                    torch.save(model.state_dict(), f)

            # early stopper: stops early after some specified number of epochs
            elif opt.early_stop != -1:
                if avg_val_loss > min_loss:
                    seq_increase += 1
                    # if the validation loss does not decrease for 3 consecutive epochs, reduce lr
                    if seq_increase == 3:
                        opt.lr = opt.lr * opt.decay_rate
                    if seq_increase == opt.early_stop:
                        break
                else:
                    seq_increase = 0

            # if the validation loss does not decrease for specified number of epochs, reduce lr
            if opt.decay_epoch != -1:
                if epoch % opt.decay_epoch == 0:
                    opt.lr = opt.lr * opt.decay_rate
                    print('new lr: {}'.format(opt.lr))

            # progress bar
            pbar.set_description(
                "Epoch {}  \t Avg. Training >> Loss: {:.4f} \t Loss RV: {:.4f} \t Loss HR: {:.4f} \t Avg. Val. Loss: {:.4f}".format(epoch, avg_loss, avg_loss_rv, avg_loss_hr, avg_val_loss))
            pbar.update(1)


def test_model(opt):
    # create fold specific dictionaries
    test_data = get_dictionary(opt.test_fold)
    # get number of  total channels
    chs = get_roi_len(opt.roi_list)

    # device CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}

    test_set = data_to_tensor(test_data, opt.roi_list, opt.percent)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, **kwargs)
    # print('hi!')

    # the network
    if opt.model == 'Bi-LSTM':
        model = BidirectionalLSTM(chs, 2048, 1)
    else:
        print('Error!')

    if opt.mode == 'test':
        model_file = '{}models/{}/saved_model_split_{}'.format(opt.out_dir, opt.uni_id, opt.train_fold)
        model.load_state_dict(torch.load(model_file))
        # count number of parameters in the model
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print('Total number of parameters: %d' % pytorch_total_params)

        # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model = model.to(device)

    avg_loss, target_rvs, target_hrs, pred_rvs, pred_hrs = test(model, device, test_loader, opt)
    #
    # # plot prediction vs output
    # plt.figure(figsize=(15.5, 5))
    #
    # n = 100
    # m = 137
    # target = target_hrs[n][:m]
    # hr = pred_hrs[n][:m]
    # thr = (target - target.mean(axis=0)) / target.std(axis=0)  # z-score normalization
    # phr = (hr - hr.mean(axis=0)) / hr.std(axis=0)  # z-score normalization
    #
    # target = target_rvs[n][:m]
    # hr = pred_rvs[n][:m]
    # trv = (target - target.mean(axis=0)) / target.std(axis=0)  # z-score normalization
    # prv = (hr - hr.mean(axis=0)) / hr.std(axis=0)  # z-score normalization
    #
    # plt.subplot(211)
    # plt.plot(np.arange(0, m), phr)
    # plt.plot(np.arange(0, m), thr)
    # plt.ylabel('hr')
    # plt.legend(['Prediction', 'Target'])
    # plt.subplot(212)
    # plt.plot(np.arange(0, m), prv)
    # plt.plot(np.arange(0, m), trv)
    # plt.ylabel('rv')
    # plt.legend(['Prediction', 'Target'])
    # plt.show()

    # Save statistics
    prediction_file = '{}results/{}/test/{}/pred_scans.csv'.format(opt.out_dir, opt.uni_id, opt.test_fold.rstrip('.txt'))
    fold_file = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/k_fold_files/' + opt.test_fold
    # fold_file = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/social_files/' + opt.test_fold

    rvp = '{}/results/{}/test/{}/rv_pred.csv'.format(opt.out_dir, opt.uni_id, opt.test_fold.rstrip('.txt'))
    rvt = '{}/results/{}/test/{}/rv_target.csv'.format(opt.out_dir, opt.uni_id, opt.test_fold.rstrip('.txt'))
    hrp = '{}/results/{}/test/{}/hr_pred.csv'.format(opt.out_dir, opt.uni_id, opt.test_fold.rstrip('.txt'))
    hrt = '{}/results/{}/test/{}/hr_target.csv'.format(opt.out_dir, opt.uni_id, opt.test_fold.rstrip('.txt'))

    os.makedirs(rvp.rstrip('rv_pred.csv'))

    with open(prediction_file, "w") as f1, open(fold_file, "r") as f2, open('nan_files.txt', "w") as f3:
        for n, line in enumerate(f2):
            id = line.split('_')[1]
            file = line.rstrip('.mat\n')
            print(n, ' ', file)
            if np.isnan(pred_rvs[n]).all():
                f3.write(file)
                f3.write('\n')
            else:
                rv_corr_coeff = sp.stats.pearsonr(pred_rvs[n][:].squeeze(), target_rvs[n][:].squeeze())
                hr_corr_coeff = sp.stats.pearsonr(pred_hrs[n][:].squeeze(), target_hrs[n][:].squeeze())

                # writing to buffer
                f1.write('{}, {}, {}, {}'.format(id, file, str(rv_corr_coeff[0]), str(hr_corr_coeff[0])))
                f1.write('\n')

                # writing to disk
                f1.flush()

                with open(rvp, "a") as file:
                    wr = csv.writer(file, delimiter=',')
                    wr.writerow(pred_rvs[n])

                with open(rvt, "a") as file:
                    wr = csv.writer(file, delimiter=',')
                    wr.writerow(target_rvs[n])

                with open(hrp, "a") as file:
                    wr = csv.writer(file, delimiter=',')
                    wr.writerow(pred_hrs[n])

                with open(hrt, "a") as file:
                    wr = csv.writer(file, delimiter=',')
                    wr.writerow(target_hrs[n])


    pass

def main():
    # pass in command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--continue_training', action='store_true', help='Continue training from saved model')
    parser.add_argument('--decay_epoch', type=int, default=-1,
                        help='Decay the learning rate after every this many epochs (-1 means no lr decay)')
    parser.add_argument('--decay_rate', type=float, default=0.05,
                        help='Rate at which the learning rate will be decayed')
    parser.add_argument('--dropout', type=float, default=0.10)
    parser.add_argument('--early_stop', type=int, default=3, help='Decide to stop early after this many epochs')
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs to train for, default=10')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0001')
    parser.add_argument('--l1', type=float, default=0.5, help='loss weighting for , default=0.0001')
    parser.add_argument('--l2', type=float, default=0.5, help='learning rate, default=0.0001')
    parser.add_argument('--mode', type=str, default='train', help='Determines whether to backpropagate or not')
    parser.add_argument('--model', type=str, default='Bi-LSTM')
    parser.add_argument('--multi', type=str, default='both')
    parser.add_argument('--out_dir', type=str, default='/home/bayrakrg/neurdy/pycharm/multi-task-physio/single-roi-models/out/', help='Path to output directory')
    parser.add_argument('--percent', type=int, default=10, help='train with top % of ROIs')
    parser.add_argument('--roi', type=str, default='regs', help='the roi label')
    parser.add_argument('--roi_list', type=str, default=['schaefer', 'tractseg', 'tian', 'aan'], help='list of rois wanted to be included')
    parser.add_argument('--test_batch', type=int, default=1, help='Decides size of each val batch')
    parser.add_argument('--test_fold', default='test_fold_0.txt', help='test_fold_k')
    parser.add_argument('--train_fold', default='train_fold_0.txt', help='train_fold_k')
    parser.add_argument('--train_batch', type=int, default=16, help='Decides size of each training batch')
    parser.add_argument('--uni_id', type=str, default='Bi-LSTM_schaefertian_roi_regs_lr_0.001_l1_0.5')
    parser.add_argument('--val_split', type=float, default=0.15, help='percentage of the split')


    opt = parser.parse_args()
    print(opt)

    if not os.path.isdir(os.path.join(opt.out_dir, 'models', opt.uni_id)):
        os.makedirs(os.path.join(opt.out_dir, 'models', opt.uni_id))
    if not os.path.isdir(os.path.join(opt.out_dir, 'results', opt.uni_id)):
        os.makedirs(os.path.join(opt.out_dir, 'results', opt.uni_id))

    if opt.mode == 'train':
        train_model(opt)
    elif opt.mode == 'test':
        test_model(opt)


if __name__ == '__main__':
    main()
