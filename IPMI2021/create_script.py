import os

fname = 'missing-training-hr.sh'
activate = 'source /home/bayrakrg/Tools/VENV/python37/bin/activate'
k_fold_path = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/k_fold_files/'
out_path = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/out/'
log_path = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/logs'
top = '#!/bin/bash'
line = 'python main.py --model={} --uni_id={} --percent={} --lr={} --l1={} --l2={} --train_fold=train_fold_{}.txt --test_fold=test_fold_{}.txt --decay_epoch={} --mode={} > {}/{}/{}.txt'

mode = ['train', 'test']
model = ['Bi-LSTM']
rois = [['schaefer', 'tractseg', 'tian', 'aan']]
lr_list = ['0.001']
percent = ['0.85']
l1_list = ['0']
# l1_list = ['0', '0.0001', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.9999', '1']
decay = ['999']

line_list = []
with open(fname, 'w') as f:
	f.write('{}\n'.format(top))
	f.write('{}\n'.format(activate))

	for roi in rois:
		for i, mo in enumerate(model):
			for lr in lr_list:
				for l1 in l1_list:
					for p in percent:
						for m in mode:
							folds = os.listdir(k_fold_path)
							for fo in folds:
								if m in fo:
									id = fo.strip('tesrainfoldx_.')
									uni_id = '{}_{}_lr_{}_l1_{}_percent_{}'.format(mo, roi[0]+roi[1]+roi[2]+roi[3], lr, l1, p)

									# create log directories
									log_path = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/logs/{}'.format(uni_id)
									if not os.path.isdir(log_path):
										os.makedirs('/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/logs/{}/train'.format(uni_id))
										os.makedirs('/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/logs/{}/test'.format(uni_id))
									if m == 'test':
										run = line.format(mo, uni_id, '0.0', lr, l1, str(round((1 - float(l1)), 4)), id, id,
														  decay[i], m, log_path, m, id)
									else:
										run = line.format(mo, uni_id, p, lr, l1, str(round((1-float(l1)), 4)), id, id,
														  decay[i], m, log_path, m, id)
									# line_list.append(run)
									f.write('{}\n'.format(run))