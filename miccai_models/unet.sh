#!/bin/bash
source /home/bayrakrg/Tools/VENV/python37/bin/activate
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0 --lr=0.000001 --l1=0 --l2=1.0 --train_fold=train_fold_2.txt --test_fold=test_fold_2.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0/train/2.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0 --lr=0.000001 --l1=0 --l2=1.0 --train_fold=train_fold_0.txt --test_fold=test_fold_0.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0/train/0.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0 --lr=0.000001 --l1=0 --l2=1.0 --train_fold=train_fold_1.txt --test_fold=test_fold_1.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0/train/1.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0 --lr=0.000001 --l1=0 --l2=1.0 --train_fold=train_fold_4.txt --test_fold=test_fold_4.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0/train/4.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0 --lr=0.000001 --l1=0 --l2=1.0 --train_fold=train_fold_3.txt --test_fold=test_fold_3.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0/train/3.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0 --lr=0.000001 --l1=0 --l2=1.0 --train_fold=train_fold_4.txt --test_fold=test_fold_4.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0/test/4.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0 --lr=0.000001 --l1=0 --l2=1.0 --train_fold=train_fold_3.txt --test_fold=test_fold_3.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0/test/3.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0 --lr=0.000001 --l1=0 --l2=1.0 --train_fold=train_fold_1.txt --test_fold=test_fold_1.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0/test/1.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0 --lr=0.000001 --l1=0 --l2=1.0 --train_fold=train_fold_2.txt --test_fold=test_fold_2.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0/test/2.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0 --lr=0.000001 --l1=0 --l2=1.0 --train_fold=train_fold_0.txt --test_fold=test_fold_0.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0/test/0.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.0001 --lr=0.000001 --l1=0.0001 --l2=0.9999 --train_fold=train_fold_2.txt --test_fold=test_fold_2.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.0001/train/2.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.0001 --lr=0.000001 --l1=0.0001 --l2=0.9999 --train_fold=train_fold_0.txt --test_fold=test_fold_0.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.0001/train/0.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.0001 --lr=0.000001 --l1=0.0001 --l2=0.9999 --train_fold=train_fold_1.txt --test_fold=test_fold_1.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.0001/train/1.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.0001 --lr=0.000001 --l1=0.0001 --l2=0.9999 --train_fold=train_fold_4.txt --test_fold=test_fold_4.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.0001/train/4.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.0001 --lr=0.000001 --l1=0.0001 --l2=0.9999 --train_fold=train_fold_3.txt --test_fold=test_fold_3.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.0001/train/3.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.0001 --lr=0.000001 --l1=0.0001 --l2=0.9999 --train_fold=train_fold_4.txt --test_fold=test_fold_4.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.0001/test/4.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.0001 --lr=0.000001 --l1=0.0001 --l2=0.9999 --train_fold=train_fold_3.txt --test_fold=test_fold_3.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.0001/test/3.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.0001 --lr=0.000001 --l1=0.0001 --l2=0.9999 --train_fold=train_fold_1.txt --test_fold=test_fold_1.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.0001/test/1.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.0001 --lr=0.000001 --l1=0.0001 --l2=0.9999 --train_fold=train_fold_2.txt --test_fold=test_fold_2.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.0001/test/2.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.0001 --lr=0.000001 --l1=0.0001 --l2=0.9999 --train_fold=train_fold_0.txt --test_fold=test_fold_0.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.0001/test/0.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.1 --lr=0.000001 --l1=0.1 --l2=0.9 --train_fold=train_fold_2.txt --test_fold=test_fold_2.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.1/train/2.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.1 --lr=0.000001 --l1=0.1 --l2=0.9 --train_fold=train_fold_0.txt --test_fold=test_fold_0.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.1/train/0.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.1 --lr=0.000001 --l1=0.1 --l2=0.9 --train_fold=train_fold_1.txt --test_fold=test_fold_1.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.1/train/1.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.1 --lr=0.000001 --l1=0.1 --l2=0.9 --train_fold=train_fold_4.txt --test_fold=test_fold_4.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.1/train/4.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.1 --lr=0.000001 --l1=0.1 --l2=0.9 --train_fold=train_fold_3.txt --test_fold=test_fold_3.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.1/train/3.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.1 --lr=0.000001 --l1=0.1 --l2=0.9 --train_fold=train_fold_4.txt --test_fold=test_fold_4.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.1/test/4.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.1 --lr=0.000001 --l1=0.1 --l2=0.9 --train_fold=train_fold_3.txt --test_fold=test_fold_3.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.1/test/3.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.1 --lr=0.000001 --l1=0.1 --l2=0.9 --train_fold=train_fold_1.txt --test_fold=test_fold_1.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.1/test/1.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.1 --lr=0.000001 --l1=0.1 --l2=0.9 --train_fold=train_fold_2.txt --test_fold=test_fold_2.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.1/test/2.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.1 --lr=0.000001 --l1=0.1 --l2=0.9 --train_fold=train_fold_0.txt --test_fold=test_fold_0.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.1/test/0.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.3 --lr=0.000001 --l1=0.3 --l2=0.7 --train_fold=train_fold_2.txt --test_fold=test_fold_2.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.3/train/2.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.3 --lr=0.000001 --l1=0.3 --l2=0.7 --train_fold=train_fold_0.txt --test_fold=test_fold_0.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.3/train/0.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.3 --lr=0.000001 --l1=0.3 --l2=0.7 --train_fold=train_fold_1.txt --test_fold=test_fold_1.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.3/train/1.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.3 --lr=0.000001 --l1=0.3 --l2=0.7 --train_fold=train_fold_4.txt --test_fold=test_fold_4.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.3/train/4.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.3 --lr=0.000001 --l1=0.3 --l2=0.7 --train_fold=train_fold_3.txt --test_fold=test_fold_3.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.3/train/3.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.3 --lr=0.000001 --l1=0.3 --l2=0.7 --train_fold=train_fold_4.txt --test_fold=test_fold_4.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.3/test/4.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.3 --lr=0.000001 --l1=0.3 --l2=0.7 --train_fold=train_fold_3.txt --test_fold=test_fold_3.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.3/test/3.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.3 --lr=0.000001 --l1=0.3 --l2=0.7 --train_fold=train_fold_1.txt --test_fold=test_fold_1.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.3/test/1.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.3 --lr=0.000001 --l1=0.3 --l2=0.7 --train_fold=train_fold_2.txt --test_fold=test_fold_2.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.3/test/2.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.3 --lr=0.000001 --l1=0.3 --l2=0.7 --train_fold=train_fold_0.txt --test_fold=test_fold_0.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.3/test/0.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.5 --lr=0.000001 --l1=0.5 --l2=0.5 --train_fold=train_fold_2.txt --test_fold=test_fold_2.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.5/train/2.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.5 --lr=0.000001 --l1=0.5 --l2=0.5 --train_fold=train_fold_0.txt --test_fold=test_fold_0.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.5/train/0.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.5 --lr=0.000001 --l1=0.5 --l2=0.5 --train_fold=train_fold_1.txt --test_fold=test_fold_1.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.5/train/1.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.5 --lr=0.000001 --l1=0.5 --l2=0.5 --train_fold=train_fold_4.txt --test_fold=test_fold_4.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.5/train/4.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.5 --lr=0.000001 --l1=0.5 --l2=0.5 --train_fold=train_fold_3.txt --test_fold=test_fold_3.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.5/train/3.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.5 --lr=0.000001 --l1=0.5 --l2=0.5 --train_fold=train_fold_4.txt --test_fold=test_fold_4.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.5/test/4.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.5 --lr=0.000001 --l1=0.5 --l2=0.5 --train_fold=train_fold_3.txt --test_fold=test_fold_3.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.5/test/3.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.5 --lr=0.000001 --l1=0.5 --l2=0.5 --train_fold=train_fold_1.txt --test_fold=test_fold_1.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.5/test/1.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.5 --lr=0.000001 --l1=0.5 --l2=0.5 --train_fold=train_fold_2.txt --test_fold=test_fold_2.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.5/test/2.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.5 --lr=0.000001 --l1=0.5 --l2=0.5 --train_fold=train_fold_0.txt --test_fold=test_fold_0.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.5/test/0.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.7 --lr=0.000001 --l1=0.7 --l2=0.3 --train_fold=train_fold_2.txt --test_fold=test_fold_2.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.7/train/2.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.7 --lr=0.000001 --l1=0.7 --l2=0.3 --train_fold=train_fold_0.txt --test_fold=test_fold_0.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.7/train/0.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.7 --lr=0.000001 --l1=0.7 --l2=0.3 --train_fold=train_fold_1.txt --test_fold=test_fold_1.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.7/train/1.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.7 --lr=0.000001 --l1=0.7 --l2=0.3 --train_fold=train_fold_4.txt --test_fold=test_fold_4.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.7/train/4.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.7 --lr=0.000001 --l1=0.7 --l2=0.3 --train_fold=train_fold_3.txt --test_fold=test_fold_3.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.7/train/3.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.7 --lr=0.000001 --l1=0.7 --l2=0.3 --train_fold=train_fold_4.txt --test_fold=test_fold_4.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.7/test/4.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.7 --lr=0.000001 --l1=0.7 --l2=0.3 --train_fold=train_fold_3.txt --test_fold=test_fold_3.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.7/test/3.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.7 --lr=0.000001 --l1=0.7 --l2=0.3 --train_fold=train_fold_1.txt --test_fold=test_fold_1.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.7/test/1.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.7 --lr=0.000001 --l1=0.7 --l2=0.3 --train_fold=train_fold_2.txt --test_fold=test_fold_2.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.7/test/2.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.7 --lr=0.000001 --l1=0.7 --l2=0.3 --train_fold=train_fold_0.txt --test_fold=test_fold_0.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.7/test/0.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9 --lr=0.000001 --l1=0.9 --l2=0.1 --train_fold=train_fold_2.txt --test_fold=test_fold_2.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9/train/2.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9 --lr=0.000001 --l1=0.9 --l2=0.1 --train_fold=train_fold_0.txt --test_fold=test_fold_0.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9/train/0.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9 --lr=0.000001 --l1=0.9 --l2=0.1 --train_fold=train_fold_1.txt --test_fold=test_fold_1.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9/train/1.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9 --lr=0.000001 --l1=0.9 --l2=0.1 --train_fold=train_fold_4.txt --test_fold=test_fold_4.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9/train/4.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9 --lr=0.000001 --l1=0.9 --l2=0.1 --train_fold=train_fold_3.txt --test_fold=test_fold_3.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9/train/3.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9 --lr=0.000001 --l1=0.9 --l2=0.1 --train_fold=train_fold_4.txt --test_fold=test_fold_4.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9/test/4.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9 --lr=0.000001 --l1=0.9 --l2=0.1 --train_fold=train_fold_3.txt --test_fold=test_fold_3.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9/test/3.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9 --lr=0.000001 --l1=0.9 --l2=0.1 --train_fold=train_fold_1.txt --test_fold=test_fold_1.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9/test/1.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9 --lr=0.000001 --l1=0.9 --l2=0.1 --train_fold=train_fold_2.txt --test_fold=test_fold_2.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9/test/2.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9 --lr=0.000001 --l1=0.9 --l2=0.1 --train_fold=train_fold_0.txt --test_fold=test_fold_0.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9/test/0.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9999 --lr=0.000001 --l1=0.9999 --l2=0.0001 --train_fold=train_fold_2.txt --test_fold=test_fold_2.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9999/train/2.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9999 --lr=0.000001 --l1=0.9999 --l2=0.0001 --train_fold=train_fold_0.txt --test_fold=test_fold_0.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9999/train/0.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9999 --lr=0.000001 --l1=0.9999 --l2=0.0001 --train_fold=train_fold_1.txt --test_fold=test_fold_1.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9999/train/1.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9999 --lr=0.000001 --l1=0.9999 --l2=0.0001 --train_fold=train_fold_4.txt --test_fold=test_fold_4.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9999/train/4.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9999 --lr=0.000001 --l1=0.9999 --l2=0.0001 --train_fold=train_fold_3.txt --test_fold=test_fold_3.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9999/train/3.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9999 --lr=0.000001 --l1=0.9999 --l2=0.0001 --train_fold=train_fold_4.txt --test_fold=test_fold_4.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9999/test/4.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9999 --lr=0.000001 --l1=0.9999 --l2=0.0001 --train_fold=train_fold_3.txt --test_fold=test_fold_3.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9999/test/3.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9999 --lr=0.000001 --l1=0.9999 --l2=0.0001 --train_fold=train_fold_1.txt --test_fold=test_fold_1.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9999/test/1.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9999 --lr=0.000001 --l1=0.9999 --l2=0.0001 --train_fold=train_fold_2.txt --test_fold=test_fold_2.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9999/test/2.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9999 --lr=0.000001 --l1=0.9999 --l2=0.0001 --train_fold=train_fold_0.txt --test_fold=test_fold_0.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_0.9999/test/0.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_1 --lr=0.000001 --l1=1 --l2=0.0 --train_fold=train_fold_2.txt --test_fold=test_fold_2.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_1/train/2.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_1 --lr=0.000001 --l1=1 --l2=0.0 --train_fold=train_fold_0.txt --test_fold=test_fold_0.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_1/train/0.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_1 --lr=0.000001 --l1=1 --l2=0.0 --train_fold=train_fold_1.txt --test_fold=test_fold_1.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_1/train/1.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_1 --lr=0.000001 --l1=1 --l2=0.0 --train_fold=train_fold_4.txt --test_fold=test_fold_4.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_1/train/4.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_1 --lr=0.000001 --l1=1 --l2=0.0 --train_fold=train_fold_3.txt --test_fold=test_fold_3.txt --decay_rate=0.05 --decay_epoch=400 --mode=train > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_1/train/3.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_1 --lr=0.000001 --l1=1 --l2=0.0 --train_fold=train_fold_4.txt --test_fold=test_fold_4.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_1/test/4.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_1 --lr=0.000001 --l1=1 --l2=0.0 --train_fold=train_fold_3.txt --test_fold=test_fold_3.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_1/test/3.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_1 --lr=0.000001 --l1=1 --l2=0.0 --train_fold=train_fold_1.txt --test_fold=test_fold_1.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_1/test/1.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_1 --lr=0.000001 --l1=1 --l2=0.0 --train_fold=train_fold_2.txt --test_fold=test_fold_2.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_1/test/2.txt
python main-mic.py --model=U-Net --uni_id=U-Net_schaefertractsegtianaan_lr_0.000001_l1_1 --lr=0.000001 --l1=1 --l2=0.0 --train_fold=train_fold_0.txt --test_fold=test_fold_0.txt --decay_rate=0.05 --decay_epoch=400 --mode=test > /home/bayrakrg/neurdy/pycharm/multi-task-physio/miccai_models/logs/U-Net_schaefertractsegtianaan_lr_0.000001_l1_1/test/0.txt
