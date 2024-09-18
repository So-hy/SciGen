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
    elif metric == "bleu_score":  # 여기서도 bleu_score 사용
        exp = "{bleu_score:.3f}-{step_count}"  # val_avg_bleu 대신 bleu_score로 수정
    else:
        raise NotImplementedError(f"seq2seq callbacks only support mover, bleu_score, got {metric}")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename=exp,
        monitor=f"{metric}",  # val_bleu_score 모니터링
        mode="max",
        save_top_k=5,
        save_last=True,
        every_n_epochs=1
    )
    return checkpoint_callback


def get_early_stopping_callback(metric, patience):
    return EarlyStopping(monitor=f"{metric}", mode="max", patience=patience, verbose=True,)

