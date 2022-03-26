import os
import time
from path import Path

import wandb
import torch
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup

from utils import *
from configs import Settings
from dataset import TrainDataset
from model import CustomModel
from trainer import train_fn, valid_fn

def get_result(oof_df):
    labels = oof_df['score'].values
    preds = oof_df['pred'].values
    score = get_score(labels, preds)
    logger.info(f'Score: {score:<.4f}')


def train_loop(folds, fold):
    
    logger.info(f"========== fold: {fold} training ==========")

    # build dataloaders
    train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)
    valid_labels = valid_folds['score'].values

    train_dataset = TrainDataset(Settings, train_folds)
    valid_dataset = TrainDataset(Settings, valid_folds)

    train_loader = DataLoader(train_dataset,
                              batch_size=Settings.batch_size,
                              shuffle=True,
                              num_workers=Settings.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=Settings.batch_size,
                              shuffle=False,
                              num_workers=Settings.num_workers, pin_memory=True, drop_last=False)

    # build model and optimizer
    model = CustomModel(Settings, config_path=None, pretrained=True)
    torch.save(model.config, OUTPUT_DIR / 'config.pth')
    model.to(device)

    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
             'lr': decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=Settings.encoder_lr, 
                                                decoder_lr=Settings.decoder_lr,
                                                weight_decay=Settings.weight_decay)

    optimizer = AdamW(optimizer_parameters, 
                      lr=Settings.encoder_lr, 
                      eps=Settings.eps, 
                      betas=Settings.betas)
    
    # lr scheduler
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=cfg.num_warmup_steps, 
                num_training_steps=num_train_steps
            )
        elif cfg.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=cfg.num_warmup_steps, 
                num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
            )
        return scheduler

    num_train_steps = int(len(train_folds) / Settings.batch_size * Settings.epochs)
    scheduler = get_scheduler(Settings, optimizer, num_train_steps)

    # epoch loops
    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    best_score = 0.
    for epoch in range(Settings.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(Settings,
                            fold, 
                            train_loader, 
                            model,
                            criterion, 
                            optimizer,
                            epoch,
                            scheduler, 
                            device, 
                            logger)
        # eval
        avg_val_loss, predictions = valid_fn(Settings,
                                             valid_loader, 
                                             model,
                                             criterion,
                                             device,
                                             logger)
        # scoring
        score = get_score(valid_labels, predictions)

        elapsed = time.time() - start_time
        logger.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        logger.info(f'Epoch {epoch+1} - Score: {score:.4f}')
    
        wandb.log({f"[fold{fold}] epoch": epoch+1, 
                    f"[fold{fold}] avg_train_loss": avg_loss, 
                    f"[fold{fold}] avg_val_loss": avg_val_loss,
                    f"[fold{fold}] score": score})

        if best_score < score:
            best_score = score
            logger.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'predictions': predictions},
                        OUTPUT_DIR / f"{Settings.model.replace('/', '-')}_fold{fold}_best.pth")

    predictions = torch.load(OUTPUT_DIR / f"{Settings.model.replace('/', '-')}_fold{fold}_best.pth", 
                             map_location=torch.device('cpu'))['predictions']

    valid_folds['pred'] = predictions

    torch.cuda.empty_cache()
    return valid_folds


if __name__ == '__main__':
    if Settings.debug:
        Settings.epochs = 2
        Settings.trn_fold = [0]
    
    wandb.login(key=Settings.wandb_api)
    wandb.init(
        project=Settings.competition,
        name=Settings.model,
        config=dataclass_to_dict(Settings),
        group=Settings.model,
        job_type='train'
    )

    INPUT_DIR = Path('/home/syoya/data/PPPM')
    OUTPUT_DIR = Path(wandb.run.dir)

    seed_everything(Settings.seed)

    logger = get_logger(Settings.competition)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load data
    train = pd.read_csv(INPUT_DIR / 'train.csv')
    test = pd.read_csv(INPUT_DIR / 'test.csv')
    submission = pd.read_csv(INPUT_DIR / 'sample_submission.csv')
    
    logger.info('train data shape: {}'.format(train.shape))
    logger.info('test data shape: {}'.format(train.shape))
    
    cpc_texts = get_cpc_texts()
    torch.save(cpc_texts, OUTPUT_DIR / "cpc_texts.pth")
    train['context_text'] = train['context'].map(cpc_texts)
    test['context_text'] = test['context'].map(cpc_texts)

    train['text'] = train['anchor'] + '[SEP]' + train['target'] + '[SEP]'  + train['context_text']
    test['text'] = test['anchor'] + '[SEP]' + test['target'] + '[SEP]'  + test['context_text']

    # cross validation split
    train['score_map'] = train['score'].map({0.00: 0, 0.25: 1, 0.50: 2, 0.75: 3, 1.00: 4})
    Fold = StratifiedKFold(n_splits=Settings.n_fold,
                           shuffle=True,
                           random_state=Settings.seed)
    for n, (train_index, val_index) in enumerate(Fold.split(train, train['score_map'])):
        train.loc[val_index, 'fold'] = int(n)
    train['fold'] = train['fold'].astype(int)

    if Settings.debug:
        train = train.sample(n=1000, random_state=0).reset_index(drop=True)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Settings.model)
    tokenizer.save_pretrained(OUTPUT_DIR / 'tokenizer/')
    Settings.tokenizer = tokenizer
    
    # define max_len
    lengths_dict = {}
    lengths = []
    for text in cpc_texts.values():
        length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
        lengths.append(length)
    lengths_dict['context_text'] = lengths

    for text_col in ['anchor', 'target']:
        lengths = []
        for text in train[text_col].fillna("").values:
            length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
            lengths.append(length)
        lengths_dict[text_col] = lengths
        
    Settings.max_len = max(lengths_dict['anchor']) + max(lengths_dict['target'])\
                    + max(lengths_dict['context_text']) + 4 # CLS + SEP + SEP + SEP
    logger.info(f"max_len: {Settings.max_len}")

    oof_df = pd.DataFrame()
    for fold in range(Settings.n_fold):
        if fold in Settings.trn_fold:
            _oof_df = train_loop(train, fold)
            oof_df = pd.concat([oof_df, _oof_df])
            logger.info(f"========== fold: {fold} result ==========")
            get_result(_oof_df)

    oof_df = oof_df.reset_index(drop=True)
    logger.info(f"========== CV ==========")
    get_result(oof_df)
    oof_df.to_pickle(OUTPUT_DIR / 'oof_df.pkl')
