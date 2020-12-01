import os

fname = 'window.sh'
activate = 'source /home/bayrakrg/Tools/VENV/python37/bin/activate'
k_fold_path = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/k_fold_files/'
log_path = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/neuroimage/log'
top = '#!/bin/bash'
line = 'python main.py --model={} --roi_clust={} --uni_id={} --l1={} --out_dir={} --train_fold=train_fold_{}.txt --test_fold=test_fold_{}.txt --mode={} > {}/{}/{}/{}.txt'

mode = ['train', 'test']
model = 'sepCONV1d'
rois = ['schaefer', 'tractseg', 'tian', 'aan']
l1_list = ['0', '0.0001', '0.1', '0.3', '0.5', '0.7', '0.9', '0.9999', '1']

line_list = []
with open(fname, 'w') as f:
	f.write('{}\n'.format(top))
	f.write('{}\n'.format(activate))

	out_dir = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/neuroimage/out/'
	for l1 in l1_list:
		for m in mode:
			folds = os.listdir(k_fold_path)
			for fo in folds:
				if m in fo:
					id = fo.strip('tesrainfoldx_.')
					uni_id = '{}_{}_l1_{}'.format(model, ''.join(rois), l1)
					run = line.format(model, ''.join(rois), uni_id, l1, out_dir, id, id, m, log_path, uni_id, m, id)
					# line_list.append(run)
					f.write('{}\n'.format(run))