import os


import torch


from mcrgan.default import _C as config
from mcrgan.trainer import MCRTrainer
from mcrgan.datasets import get_dataloader
from mcrgan.models import get_models
from main import _to_yaml


def run_unsup():
    
    log_dir =  '/Users/ryanchankh/LDR/saved_models/LDR_multi_mode2_mini_dcgan/config.yaml'
    _to_yaml(config, os.path.join(log_dir, 'config.yaml'))
    
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    dataloader, dataset = get_dataloader(
        data_name=config.DATA.DATASET,
        root=config.DATA.ROOT,
        batch_size=config.TRAIN.BATCH_SIZE,
        num_workers=config.CUDNN.WORKERS
    )
    
    # Load model and checkpoint
    netD, netG = get_models(config.DATA.DATASET, device)
    netD_ckpt_dir = os.path.join(log_dir, 'checkpoints', 'netD')
    netD_ckpt_file = os.path.join(netD_ckpt_dir, 'netD_45000_steps.pth')
    netD.load_state_dict(torch.load(netD_ckpt_file))

    # Start training
    
    
    
if __name__ == '__main__':
    run_unsup()
    
    
    