from speedrun.logging.wandb import SweepRunner
from ctt.training import CTTTrainer


if __name__ == '__main__':
    SweepRunner(CTTTrainer).run()
