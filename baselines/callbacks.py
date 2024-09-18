import logging
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only


def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


logger = logging.getLogger(__name__)



def get_checkpoint_callback(output_dir, metric):
    """Saves the best model by validation metric score."""
    if metric == "mover":
        exp = "{val_mover:.2f}-{step_count}"
    elif metric == "mover_median":
        exp = "{val_mover_median:.4f}-{step_count}"
    elif metric == "bleu":
        exp = "{val_avg_bleu:.3f}-{step_count}"
    else:
        raise NotImplementedError(f"seq2seq callbacks only support mover, bleu, got {metric}")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,  # 최신 버전에서는 filepath 대신 dirpath 사용
        filename=exp,
        monitor=f"val_{metric}",
        mode="max",
        save_top_k=5,
        save_last=True,  # 마지막 체크포인트도 저장
        every_n_epochs=1  # 매 epoch마다 저장
    )
    return checkpoint_callback


def get_early_stopping_callback(metric, patience):
    return EarlyStopping(monitor=f"val_{metric}", mode="max", patience=patience, verbose=True,)

