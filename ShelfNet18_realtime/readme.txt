to evaluate run 
sudo python3 evaluate.py
(download model_final.pth weights and put it in res folder)

to train
CUDA_VISIBLE_DEVICES=0 sudo python3 -m torch.distributed.launch --nproc_per_node=2 train.py

I have changed the evaluate.py,train.py,cityscpes_info.json scripts (little)
To get the error try running the train command.To change the number of iterations go to train.py and change.
I have mixed the idd and iisc data and uploaded in data folder.The log.txt has the error details






