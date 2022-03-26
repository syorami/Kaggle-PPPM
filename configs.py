from dataclasses import dataclass


@dataclass
class Settings:
    wandb_api = 'local-f99f9efd3491da2b14d0ac6d8e8256f3dfb55d60'
    competition = 'PPPM'
    debug = False
    apex = True
    print_freq = 100
    num_workers = 4
    model = "microsoft/deberta-v3-large"
    scheduler = 'cosine'  # ['linear', 'cosine']
    batch_scheduler = True
    num_cycles = 0.5
    num_warmup_steps = 0
    epochs = 4
    encoder_lr = 2e-5
    decoder_lr = 2e-5
    min_lr = 1e-6
    eps = 1e-6
    betas = (0.9, 0.999)
    batch_size = 16
    fc_dropout = 0.2
    target_size = 1
    max_len = 512
    weight_decay = 0.01
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    seed = 42
    n_fold = 4
    trn_fold = [0, 1, 2, 3]
    train = True
