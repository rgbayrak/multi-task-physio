import os

fname = 'hr-task.sh'
activate = 'source /home/bayrakrg/Tools/VENV/python37/bin/activate'
k_fold_path = '/home/bayrakrg/neurdy/pycharm/HR/k_fold_files/'
out_path = '/home/bayrakrg/neurdy/pycharm/HR/out/'
log_path = '/home/bayrakrg/neurdy/pycharm/HR/logs'
top = '#!/bin/bash'
line = 'python main.py --model={} --uni_id={} --lr={} --train_fold=train_fold_{}.txt --test_fold=test_fold_{}.txt --decay_epoch={} --mode={} > {}/{}/{}.txt'

mode = ['train', 'test']
model = ['Bi-LSTM']
rois = [['findlab', 'wmcsf']]
lr_list = ['0.001', '0.0001', '0.00001']
decay = ['10']

line_list = []
with open(fname, 'w') as f:
	f.write('{}\n'.format(top))
	f.write('{}\n'.format(activate))

	for roi in rois:
		for i, mo in enumerate(model):
			for lr in lr_list:
				for m in mode:
					folds = os.listdir(k_fold_path)
					for fo in folds:
						if m in fo:
							id = fo.strip('tesrainfoldx_.')
							uni_id = '{}_{}_lr_{}'.format(mo, roi[0]+roi[1], lr)

							# create log directories
							log_path = '/home/bayrakrg/neurdy/pycharm/HR/logs/{}'.format(uni_id)
							if not os.path.isdir(log_path):
								os.makedirs('/home/bayrakrg/neurdy/pycharm/HR/logs/{}/train'.format(uni_id))
								os.makedirs('/home/bayrakrg/neurdy/pycharm/HR/logs/{}/test'.format(uni_id))

							run = line.format(mo, uni_id, lr, id, id, decay[i], m, log_path, m, id)
							# line_list.append(run)
							f.write('{}\n'.format(run))