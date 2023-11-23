python main.py --dataname searchsnippets_trans_subst_10  --num_classes 8 --classes 8 --M 74 --epsion 0.1 --pre_step 600 --temperature 1 --start 0 --reg2 0.1  --logalpha 10 --objective $1
wait
python main.py --dataname agnewsdataraw-8000_trans_subst_10 --num_classes 4 --classes 4 --M 110 --epsion 0.1 --pre_step 600 --temperature 1 --start 0 --reg2 1 --logalpha 10 --objective $1
wait
python main.py --dataname stackoverflow_trans_subst_10 --num_classes 20 --classes 20 --M 50 --epsion 0.1 --pre_step -1 --temperature 1 --start 0 --reg2 1 --logalpha 10 --objective $1
wait
python main.py --dataname biomedical_trans_subst_10 --num_classes 20 --classes 20 --M 50 --epsion 0.1 --pre_step -1 --temperature 1 --start 0  --reg2 1 --logalpha 10 --objective $1
wait
python main.py --dataname TS_trans_subst_10 --num_classes 152 --classes 152 --M 82 --epsion 0.1 --pre_step 600 --temperature 1 --start 0  --reg2 0.1 --logalpha 100 --objective $1
wait
python main.py --dataname T_trans_subst_10 --num_classes 152 --classes 152 --M 82 --epsion 0.1 --pre_step 600 --temperature 1 --start 0  --reg2 0.1 --logalpha 100 --objective $1
wait
python main.py --dataname S_trans_subst_10 --num_classes 152 --classes 152 --M 82 --epsion 0.1 --pre_step 600 --temperature 1 --start 0 --reg2 0.1 --logalpha 100 --objective $1
wait
python main.py --dataname tweet-original-order_trans_subst_10 --num_classes 89 --classes 89 --M 110  --epsion 0.1 --pre_step -1 --temperature 1 --start 0 --reg2 0.1 --logalpha 100 --objective $1