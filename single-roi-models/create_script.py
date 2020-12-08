import os
import numpy as np

fname = 'select-roi.sh'
activate = 'source /home/bayrakrg/Tools/VENV/python37/bin/activate'
k_fold_path = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/k_fold_files/'
out_path = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/out/'
log_path = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/logs'
top = '#!/bin/bash'
line = 'python main-single-roi.py --model={} --uni_id={} --lr={} --l1={} --l2={} --percent={} --train_fold=train_fold_{}.txt --test_fold=test_fold_{}.txt --decay_epoch={} --mode={} > {}/{}/{}.txt'

mode = ['train', 'test']
model = ['Bi-LSTM']
rois = ['schaefer', 'tractseg', 'tian', 'aan']
percent = ['10', '5', '1', '25', '50', '100']
lr_list = ['0.001']
l1_list = ['0.5']
decay = ['2']

line_list = []
with open(fname, 'w') as f:
	f.write('{}\n'.format(top))
	f.write('{}\n'.format(activate))

	for p in percent:
		for i, mo in enumerate(model):
			for lr in lr_list:
				for l1 in l1_list:
					for m in mode:
						folds = os.listdir(k_fold_path)
						for fo in folds:
							if m in fo:
								id = fo.strip('tesrainfoldx_.')
								uni_id = '{}_{}_{}%_lr_{}_l1_{}'.format(mo, rois[0]+rois[1], p, lr, l1)

								# create log directories
								log_path = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/logs/{}'.format(uni_id)
								if not os.path.isdir(log_path):
									os.makedirs('/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/logs/{}/train'.format(uni_id))
									os.makedirs('/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/logs/{}/test'.format(uni_id))

								run = line.format(mo, uni_id, lr, l1, str(round((1-float(l1)), 4)), p, id, id, decay[i], m, log_path, m, id)
								# line_list.append(run)
								f.write('{}\n'.format(run))