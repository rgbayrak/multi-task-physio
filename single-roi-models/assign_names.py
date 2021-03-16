import os
fn = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/single-roi-models/out/results/Bi-LSTM_tractsegaan_roi_'
fn1 = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/single-roi-models/out/results/Bi-LSTM_tractseg_roi_'
fn2 = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/single-roi-models/out/results/Bi-LSTM_aan_roi_'
with open('tractsegaan_roi.txt', 'r') as f:
    fname = f.readlines()

for i in range(len(fname)):
    fname[i] = [str(x) for x in fname[i].strip('\n').split(',')]

dict = {}
for n, j in enumerate(fname):
    if n != 4:
        if n < 72:
            os.rename(fn + str(n) + '_lr_0.001_l1_0.5', fn1 + j[0])
            print(n, j[0])
        else:
            os.rename(fn + str(n) + '_lr_0.001_l1_0.5', fn2 + j[0])


pass
