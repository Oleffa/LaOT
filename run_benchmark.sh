# An example of how to use train LaOT
python train_AE.py --benchmark --hid1 256 --hid2 256 --d1 'amazon' --d2 'webcam' --f1 'GoogleNet1024' --f2 'CaffeNet4096' --a 3e-5 --z 256 --lr 5e-4 --b1 958 --b2 295 --torch_seed 0 --np_seed 0 --e 25
