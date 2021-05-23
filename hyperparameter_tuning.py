import os
from itertools import product
#import random

OPTIM_NAME = "LaRSPaG"
USE_LARGERNET = False
LOG_ON = False
LR_DECAY_DEG = 2
print(f"Log on argumet is {LOG_ON}\n")

COMMANDS = {
    "SGD": lambda x: f"python train.py --batch-size {x[0]} --lr {x[1]} --optimizer {OPTIM_NAME}",
    "LARS": lambda x: f"python train.py --batch-size {x[0]} --lr {x[1]} --optimizer {OPTIM_NAME} --eta {x[2]}",
    "GradClip": lambda x: f"python train.py --batch-size {x[0]} --lr {x[1]} --optimizer {OPTIM_NAME} --clip {x[2]}",
    "LGC": lambda x: f"python train.py --batch-size {x[0]} --lr {x[1]} --optimizer {OPTIM_NAME} --clip {x[2]}",
    "LaRSPaG": lambda x: f"python train.py --batch-size {x[0]} --lr {x[1]} --optimizer {OPTIM_NAME} --eta {x[2]} --clip {x[3]}",
}
GENERATORS = {
    "SGD": product(range(3,4+1), range(-1,2+1)),
    "LARS": product(range(3,4+1), range(-1,2+1), range(-3,2+1)),
    "GradClip": product(range(3,4+1), range(-1,2+1), range(-3,2+1)),
    "LGC": product(range(3,4+1), range(-1,2+1), range(-3,2+1)),
    "LaRSPaG": product(range(3,4+1), range(-1,2+1), range(-3,2+1), range(-3,2+1)),
}
EXP10 = lambda x: [10 ** p  for p in x]


def main():
    for params in GENERATORS[OPTIM_NAME]:
        command = COMMANDS[OPTIM_NAME](EXP10(params))
        if USE_LARGERNET:
            command += " --use-largernet"
        if LOG_ON:
            command += " --log-file-on"
        if LR_DECAY_DEG != 2:
            command += f" --lr-decay-degree {LR_DECAY_DEG}"
        if os.system(command) != 0:
            break

if __name__ == '__main__':
    main()
