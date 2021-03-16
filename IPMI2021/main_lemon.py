import torch
import torchvision.transforms as transforms
import argparse
from torch.utils.data import DataLoader
# from data_loader import *
from data_loader_lemon import *
from trainer_lemon import *
from model import *
from tqdm import tqdm
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
import csv
import os


def test_model(opt):
    # create fold specific dictionaries
    test_data = get_dictionary(opt.test_fold)
    # get number of  total channels
    chs = get_roi_len(opt.roi_list)

    # device CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}

    test_set = data_to_tensor(test_data, opt.roi_list)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, **kwargs)
    # print('hi!')

    # the network
    if opt.model == 'Bi-LSTM':
        model = BidirectionalLSTM(chs, 2000, 1)
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

    avg_loss, target_rvs, pred_rvs = test(model, device, test_loader, opt)
    #
    # plot prediction vs output
    plt.figure(figsize=(15.5, 5))

    # n = 1
    # m = 633
    # # target = target_hrs[n][:m]
    # # hr = pred_hrs[n][:m]
    # # thr = (target - target.mean(axis=0)) / target.std(axis=0)  # z-score normalization
    # # phr = (hr - hr.mean(axis=0)) / hr.std(axis=0)  # z-score normalization
    #
    # target = target_rvs[n][:m]
    # rv = pred_rvs[n][:m]
    # trv = (target - target.mean(axis=0)) / target.std(axis=0)  # z-score normalization
    # prv = (rv - rv.mean(axis=0)) / rv.std(axis=0)  # z-score normalization
    #
    # # plt.subplot(211)
    # # plt.plot(np.arange(0, m), phr)
    # # plt.plot(np.arange(0, m), thr)
    # # plt.ylabel('hr')
    # # plt.legend(['Prediction', 'Target'])
    # plt.subplot(712)
    # plt.plot(np.arange(0, m), prv)
    # plt.plot(np.arange(0, m), trv)
    # plt.ylabel('rv')
    # # plt.legend(['Prediction', 'Target'])
    # plt.show()

    # Save statistics
    prediction_file = '{}lemon-results/{}/test/{}/pred_scans.csv'.format(opt.out_dir, opt.uni_id, opt.test_fold.rstrip('.txt'))
    fold_file = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/lemon_files/' + opt.test_fold
    # fold_file = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/social_files/' + opt.test_fold

    rvp = '{}/lemon-results/{}/test/{}/rv_pred.csv'.format(opt.out_dir, opt.uni_id, opt.test_fold.rstrip('.txt'))
    rvt = '{}/lemon-results/{}/test/{}/rv_target.csv'.format(opt.out_dir, opt.uni_id, opt.test_fold.rstrip('.txt'))

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

                # writing to buffer
                f1.write('{}, {}, {}'.format(id, file, str(rv_corr_coeff[0])))
                f1.write('\n')

                # writing to disk
                f1.flush()

                with open(rvp, "a") as file:
                    wr = csv.writer(file, delimiter=',')
                    wr.writerow(pred_rvs[n])

                with open(rvt, "a") as file:
                    wr = csv.writer(file, delimiter=',')
                    wr.writerow(target_rvs[n])

    pass

def main():
    # pass in command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='Bi-LSTM')
    parser.add_argument('--multi', type=str, default='both')
    parser.add_argument('--uni_id', type=str, default='Bi-LSTM_all4_lr_0.001_l1_0.5')
    parser.add_argument('--epoch', type=int, default=15, help='number of epochs to train for, default=10')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0001')
    parser.add_argument('--l1', type=float, default=0.5, help='loss weighting for , default=0.0001')
    parser.add_argument('--l2', type=float, default=0.5, help='learning rate, default=0.0001')
    parser.add_argument('--test_fold', default='test_fold_0.txt', help='test_fold_k')
    parser.add_argument('--train_fold', default='train_fold_0.txt', help='train_fold_k')
    parser.add_argument('--val_split', type=float, default=0.15, help='percentage of the split')

    parser.add_argument('--out_dir', type=str, default='/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/out_lemon/', help='Path to output directory')
    parser.add_argument('--roi_list', type=str, default=['schaefer', 'tractseg', 'tian', 'aan'], help='list of rois wanted to be included')
    parser.add_argument('--mode', type=str, default='test', help='Determines whether to backpropagate or not')
    parser.add_argument('--train_batch', type=int, default=16, help='Decides size of each training batch')
    parser.add_argument('--test_batch', type=int, default=1, help='Decides size of each val batch')
    parser.add_argument('--decay_rate', type=float, default=0.5, help='Rate at which the learning rate will be decayed')
    parser.add_argument('--decay_epoch', type=int, default=1, help='Decay the learning rate after every this many epochs (-1 means no lr decay)')
    parser.add_argument('--dropout', type=float, default=0.10, help='Continue training from saved model')
    parser.add_argument('--early_stop', type=int, default=5, help='Decide to stop early after this many epochs in which the validation loss increases (-1 means no early stopping)')
    parser.add_argument('--continue_training', action='store_true', help='Continue training from saved model')

    opt = parser.parse_args()
    print(opt)

    if not os.path.isdir(os.path.join(opt.out_dir, 'models', opt.uni_id)):
        os.makedirs(os.path.join(opt.out_dir, 'models', opt.uni_id))
    if not os.path.isdir(os.path.join(opt.out_dir, 'results', opt.uni_id)):
        os.makedirs(os.path.join(opt.out_dir, 'results', opt.uni_id))

    test_model(opt)


if __name__ == '__main__':
    main()
