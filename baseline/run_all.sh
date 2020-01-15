#!/bin/sh

# basic
python main.py -d -u
# with weights initialization
python main.py -m "/srv/nfs/coco/jbedziechowska/DCASE2019_task4/dcase2019_t4_mdl" -d -u --skip_rnn --skip_dense

# sort class count
python main.py -d -u -o
python main.py -m "/srv/nfs/coco/jbedziechowska/DCASE2019_task4/dcase2019_t4_mdl" -d -u -o --skip_rnn --skip_dense

python main.py -d -u --sort_overlap
python main.py -m "/srv/nfs/coco/jbedziechowska/DCASE2019_task4/dcase2019_t4_mdl" -d -u --skip_rnn --skip_dense --sort_overlap

python main.py -d -u --sort_class
python main.py -m "/srv/nfs/coco/jbedziechowska/DCASE2019_task4/dcase2019_t4_mdl" -d -u --skip_rnn --skip_dense --sort_class

python main.py -d -u --snr
python main.py -m "/srv/nfs/coco/jbedziechowska/DCASE2019_task4/dcase2019_t4_mdl" -d -u --skip_rnn --skip_dense --snr

# flatness
python main.py -d -u -f
python main.py -m "/srv/nfs/coco/jbedziechowska/DCASE2019_task4/dcase2019_t4_mdl" -d -u -f --skip_rnn --skip_dense



