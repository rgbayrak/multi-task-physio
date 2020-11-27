import os
import numpy as np

fname = 'single-roi.sh'
activate = 'source /home/bayrakrg/Tools/VENV/python37/bin/activate'
k_fold_path = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/k_fold_files/'
out_path = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/out/'
log_path = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/logs'
top = '#!/bin/bash'
line = 'python main-single-roi.py --model={} --uni_id={} --lr={} --l1={} --l2={} --roi={} --train_fold=train_fold_{}.txt --test_fold=test_fold_{}.txt --decay_epoch={} --mode={} > {}/{}/{}.txt'

mode = ['train', 'test']
model = ['Bi-LSTM']
fold = ['0']
rois = ['tractseg', 'aan']
roi = np.arange(81) # sum(number of rois)
lr_list = ['0.001']
l1_list = ['0.5']
decay = ['2']

line_list = []
with open(fname, 'w') as f:
	f.write('{}\n'.format(top))
	f.write('{}\n'.format(activate))

	for r in roi:
		for i, mo in enumerate(model):
			for lr in lr_list:
				for l1 in l1_list:
					for m in mode:
						folds = os.listdir(k_fold_path)
						for fo in folds:
							if m in fo and fold[0] in fo:
								id = fo.strip('tesrainfoldx_.')
								uni_id = '{}_{}_roi_{}_lr_{}_l1_{}'.format(mo, rois[0]+rois[1], r, lr, l1)

								# create log directories
								log_path = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/logs/{}'.format(uni_id)
								if not os.path.isdir(log_path):
									os.makedirs('/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/logs/{}/train'.format(uni_id))
									os.makedirs('/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/logs/{}/test'.format(uni_id))

								run = line.format(mo, uni_id, lr, l1, str(round((1-float(l1)), 4)), r, id, id, decay[i], m, log_path, m, id)
								# line_list.append(run)
								f.write('{}\n'.format(run))