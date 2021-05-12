import os
import time
from train import log_maker

OPTIM_NAME = "SGD"
logger = log_maker(None, print_hyperparam=False)

for bs_exp in range(1,5):
    for lr_exp in range(-4, 3):
        bs = 10 ** bs_exp
        lr = 10 ** lr_exp
        try:
            os.system(f"python train.py --batch-size {bs} --lr {lr} --optimizer {OPTIM_NAME}")
        except Exception as e:
            logger.info("!!! Exception occured !!!")
            logger.info(e)