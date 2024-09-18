import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import sys

from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelWithLMHead,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

MODEL_MODES = {
    "base": AutoModel,
    "sequence-classification": AutoModelForSequenceClassification,
    "question-answering": AutoModelForQuestionAnswering,
    "pretraining": AutoModelForPreTraining,
    "token-classification": AutoModelForTokenClassification,
    "language-modeling": AutoModelWithLMHead,
}

def set_seed(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)  # 여러 GPU 사용 시 시드 설정

class BaseTransformer(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace, num_labels=None, mode="base", **config_kwargs):
        super().__init__()
        # 모든 하이퍼파라미터 저장
        self.save_hyperparameters()

        # 안전하게 하이퍼파라미터에 접근
        cache_dir = getattr(self.hparams, 'cache_dir', None)
        config_name = getattr(self.hparams, 'config_name', None)
        tokenizer_name = getattr(self.hparams, 'tokenizer_name', None)
        model_name_or_path = self.hparams.model_name_or_path

        # AutoConfig 설정
        self.config = AutoConfig.from_pretrained(
            config_name if config_name else model_name_or_path,
            **({"num_labels": num_labels} if num_labels is not None else {}),
            cache_dir=cache_dir,
            **config_kwargs,
        )

        # AutoTokenizer 설정
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name if tokenizer_name else model_name_or_path,
            cache_dir=cache_dir,
        )

        # 새로운 토큰 추가
        new_tokens = ['[R]', '[C]', '[CAP]']
        num_added_toks = self.tokenizer.add_tokens(new_tokens)
        self.model = MODEL_MODES[mode].from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=self.config,
            cache_dir=cache_dir,
        )
        logger.info('We have added %s tokens', num_added_toks)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def train_dataloader(self):
        train_batch_size = getattr(self.hparams, 'train_batch_size', 32)  # 기본값 설정
        dataloader = self.load_dataset("train", train_batch_size)
        t_total = (
            (len(dataloader.dataset) // (train_batch_size * max(1, self.hparams.n_gpu)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}".format(key, str(metrics[key])))

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        logger.info("***** Test results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log and save results to file
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}".format(key, str(metrics[key])))
                        writer.write("{} = {}".format(key, str(metrics[key])))

def add_generic_args(parser, root_dir):
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']." 
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--n_tpu_cores", type=int, default=0)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

def generic_train(model: BaseTransformer, args: argparse.Namespace, early_stopping_callback=None, checkpoint_callback=None):
    set_seed(args)

    if not checkpoint_callback:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=args.output_dir, monitor="val_loss", mode="min", save_top_k=-1, save_last=True
        )

    callbacks = [LoggingCallback(), checkpoint_callback]
    if early_stopping_callback:
        callbacks.append(early_stopping_callback)

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        gradient_clip_val=args.max_grad_norm,
        callbacks=callbacks,
        log_every_n_steps=1,
        num_sanity_val_steps=4,
        reload_dataloaders_every_n_epochs=True
    )

    # 혼합 정밀도 설정
    if args.fp16:
        train_params["precision"] = 16  # 최신 PyTorch Lightning에서는 'precision'으로 설정

    # TPU 설정
    if args.n_tpu_cores > 0:
        train_params["tpu_cores"] = args.n_tpu_cores

    # 멀티 GPU 설정
    if args.n_gpu > 1:
        train_params["strategy"] = "ddp"  # 멀티 GPU 학습을 위한 DDP 전략

    trainer = pl.Trainer(**train_params)

    # 모델 학습
    if args.do_train:
        trainer.fit(model)

    return trainer
