from _csv import reader, writer

import numpy as np
import torch
import torchvision.transforms as transforms
import argparse
from torch.utils.data import DataLoader
from data_loader import *
from trainer import *
from model import *
from tqdm import tqdm
import scipy as sp
import csv
from torchsummary import summary


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
    summary(net, input_size=(497, 32))


def train_model(opt):
    # device CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

    # create fold specific dictionaries
    train_data = get_dictionary(opt.train_fold)
    keys = list(train_data.keys())

    val_split = round(len(train_data) * opt.val_split)
    val_data = {}
    for i in range(val_split):
        idx = random.randint(0, len(keys) - 1)
        val_data[keys[idx]] = train_data[keys[idx]]
        del train_data[keys[idx]]
        del keys[idx]

    # load the train/val data as tensor
    train_set = data_to_tensor(train_data, opt)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=opt.train_batch, shuffle=True, **kwargs)
    #
    val_set = val_to_tensor(val_data, opt)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=1, **kwargs)

    # the network

    if opt.model == 'sepCONV1d':
        model = sepCONV1d(497, 1, opt)
    else:
        print('Error!')

    model = model.to(device)

    # load optimizer
    optim = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # save average loss throughout training
    train_loss_file = '{}/results/{}/train_loss_split_{}'.format(opt.out_dir, opt.uni_id, opt.train_fold)
    f = open(train_loss_file, 'w')
    f.close()
    validate_loss_file = '{}/results/{}/validate_loss_split_{}'.format(opt.out_dir, opt.uni_id, opt.train_fold)
    f = open(validate_loss_file, 'w')
    f.close()

    fold_info = (opt.train_fold).rsplit('.')
    model_file = '{}/models/{}/saved_model_split_{}'.format(opt.out_dir, opt.uni_id, fold_info[0])

    seq_increase = 0
    min_loss = 10000

    # finetuning or transfer learning
    if opt.continue_training:
        model.load_state_dict(torch.load(model_file))
        model = model.to(device)

    with tqdm(total=opt.epoch) as pbar:
        for epoch in range(1, opt.epoch + 1):

            # avg_loss is for that epoch, tloss is an array of all avg_losses
            avg_loss, tloss = train(model, device, train_loader, optim, opt)

            avg_val_loss, target_rvs, target_hrs, pred_rvs, pred_hrs = test(model, device, val_loader, opt)

            # save model or stop training
            if avg_val_loss < min_loss:
                min_loss = avg_val_loss
                with open(model_file, 'wb') as f:
                    torch.save(model.state_dict(), f)

            # if epoch == 5:
            #     with open(model_file, 'wb') as f:
            #         torch.save(model.state_dict(), f)

            # early stopper
            elif opt.early_stop != -1:
                if avg_val_loss > min_loss:
                    seq_increase += 1
                    # decay rate
                    if seq_increase == opt.decay_cycle:
                        opt.lr = opt.lr * opt.decay_rate
                        print('new lr: {}'.format(opt.lr))

                    if seq_increase == opt.early_stop:
                        break
                else:
                    seq_increase = 0

            # progress bar
            pbar.set_description(
                # "Epoch {}  \t Avg. Loss: {:.4f}".format(epoch, avg_loss))
                "Epoch {}  \t Avg. Loss: {:.4f}    \t Avg. Val. Loss: {:.4f}".format(epoch, avg_loss, avg_val_loss))
            pbar.update(1)

    # look at training vs. validation loss
    # x = range(len(tloss))
    # plt.figure()
    # plt.plot(x, tloss, x, vloss)
    # plt.ylabel('Cosine Similarity')
    # plt.xlabel('# of Epochs')
    # plt.legend(['Train Loss', 'Validation Loss'])
    # plt.show()

    # # visualize sample prediction
    # plt.plot(range(0, opt.epoch), p_test)
    # plt.plot(range(0, opt.epoch), t_test)
    # plt.legend(['Prediction', 'Target'])
    # plt.show()

def test_model(opt):
    # create fold specific dictionaries
    test_data = get_dictionary(opt.test_fold)

    # device CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}

    test_set = test_to_tensor(test_data, opt)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, **kwargs)
    # print('hi!')

    # the network
    if opt.model == 'sepCONV1d':
        model = sepCONV1d(497, 1, opt)
    else:
        print('Error!')

    if opt.mode == 'test':
        fold_info = (opt.train_fold).rsplit('.')
        model_file = '{}/models/{}/saved_model_split_{}'.format(opt.out_dir, opt.uni_id, fold_info[0])
        model.load_state_dict(torch.load(model_file))
    model = model.to(device)

    avg_loss, target_rvs, pred_rvs, target_hrs, pred_hrs = test(model, device, test_loader, opt)

    # Save statistics
    prediction_file = '{}rresults/{}/test/{}/pred_scans.csv'.format(opt.out_dir, opt.uni_id, opt.test_fold.rstrip('.txt'))
    fold_file = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/k_fold_files/' + opt.test_fold
    # fold_file = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/social_files/' + opt.test_fold

    rvp = '{}/rresults/{}/test/{}/rv_pred.csv'.format(opt.out_dir, opt.uni_id, opt.test_fold.rstrip('.txt'))
    rvt = '{}/rresults/{}/test/{}/rv_target.csv'.format(opt.out_dir, opt.uni_id, opt.test_fold.rstrip('.txt'))
    hrp = '{}/rresults/{}/test/{}/hr_pred.csv'.format(opt.out_dir, opt.uni_id, opt.test_fold.rstrip('.txt'))
    hrt = '{}/rresults/{}/test/{}/hr_target.csv'.format(opt.out_dir, opt.uni_id, opt.test_fold.rstrip('.txt'))

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

    return targets, preds


def main():
    # pass in command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='sepCONV1d')
    parser.add_argument('--uni_id', type=str, default='SepCONV1d_all4', help='change at every new parameter')
    parser.add_argument('--window_size', type=int, default='32', help='sliding window size')
    parser.add_argument('--epoch', type=int, default=5, help='number of epochs to train for, default=10')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.000001')
    parser.add_argument('--l1', type=float, default=0.5, help='loss weighting for , default=0.0001')
    parser.add_argument('--roi_list', default=['schaefer', 'tractseg', 'tian', 'aan'], help='clust10 | clust42 | findlab92')
    parser.add_argument('--train_fold', default='train_fold_0.txt', help='train_fold_k')
    parser.add_argument('--test_fold', default='test_fold_0.txt', help='test_fold_k')
    parser.add_argument('--val_split', type=float, default=0.15, help='percentage of the split')

    parser.add_argument('--out_dir', type=str, default='/home/bayrakrg/neurdy/pycharm/multi-task-physio/neuroimage/', help='Path to output directory')
    parser.add_argument('--mode', type=str, default='train', help='Determines whether to backpropagate or not')
    parser.add_argument('--train_batch', type=int, default=512, help='Decides size of each training batch')
    parser.add_argument('--test_batch', type=int, default=1, help='Decides size of each val batch')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='Rate at which the learning rate will be decayed')
    parser.add_argument('--decay_cycle', type=int, default=2, help='Cycle at which the learning rate will be decayed')
    parser.add_argument('--decay_epoch', type=int, default=-1, help='Decay the learning rate after every this many epochs (-1 means no lr decay)')
    parser.add_argument('--dropout', type=float, default=0.10, help='Continue training from saved model')
    parser.add_argument('--early_stop', type=int, default=-1, help='Decide to stop early after this many epochs in which the validation loss increases (-1 means no early stopping)')
    parser.add_argument('--continue_training', action='store_true', help='Continue training from saved model')

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
