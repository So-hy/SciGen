import argparse
import glob
import logging
import os
import time
from typing import List
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
import sys
from lightning_base import BaseTransformer, add_generic_args, generic_train, get_linear_schedule_with_warmup
from callbacks import get_checkpoint_callback, get_early_stopping_callback
from utils import convert_text, eval_bleu_sents, eval_sacre_bleu, eval_mover_score, eval_bleu
from utils import Table2textDataset as AgendaDataset

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

class SummarizationTrainer(BaseTransformer):

    mode = "language-modeling"
    val_metric = "mover"

    def __init__(self, hparams):
        super().__init__(hparams, num_labels=None, mode=self.mode)

        self.metrics_save_path = Path(self.hparams.output_dir) / "metrics.json"
        self.hparams_save_path = Path(self.hparams.output_dir) / "hparams.pkl"
        self.step_count = 0
        self.metrics = defaultdict(list)

        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            max_target_length=self.hparams.max_target_length,
        )
        self.count_valid_epoch = 0

        logger.info("parameters %s", hparams)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, lm_labels=None):
        return self.model(
            input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, lm_labels=lm_labels,
        )

    def _step(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = batch["source_ids"], batch["source_mask"], batch["target_ids"]
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone()
        lm_labels[y[:, 1:] == pad_token_id] = -100
        outputs = self(source_ids, attention_mask=source_mask, decoder_input_ids=y_ids, lm_labels=lm_labels,)

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = AgendaDataset.trim_seq2seq_batch(batch, pad_token_id)
        generated_ids = self.model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            num_beams=5,
            max_length=512,
            length_penalty=5.0,
            early_stopping=True,
            use_cache=True,
        )
        preds = [
            self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for g in generated_ids
        ]
        target = [self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
        loss = self._step(batch)
        return {"val_loss": loss, "preds": preds, "target": target}

    def check_validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = AgendaDataset.trim_seq2seq_batch(batch, pad_token_id)
        generated_ids = self.model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            num_beams=5,
            max_length=512,
            length_penalty=5.0,
            early_stopping=True,
            use_cache=True,
        )
        preds = [
            self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for g in generated_ids
        ]
        target = [self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
        loss = self._step(batch)
        return {"val_loss": loss, "preds": preds, "target": target}

    def test_epoch_end(self, outputs):
        if "preds" in outputs[0]:
            output_test_predictions_file = os.path.join(self.hparams.output_dir, "test_predictions_" +
                                                        str(self.count_valid_epoch) + ".txt")
            output_test_targets_file = os.path.join(self.hparams.output_dir, "test_targets_" +
                                                        str(self.count_valid_epoch) + ".txt")
            with open(output_test_predictions_file, "w") as p_writer, open(output_test_targets_file, "w") as t_writer:
                for output_batch in outputs:
                    p_writer.writelines(convert_text(s) + "\n" for s in output_batch["preds"])
                    t_writer.writelines(convert_text(s) + "\n" for s in output_batch["target"])
            bleu_info = eval_sacre_bleu(output_test_targets_file, output_test_predictions_file)
            moverScore = eval_mover_score(output_test_targets_file, output_test_predictions_file)

            logger.info("valid epoch: %s", self.count_valid_epoch)
            logger.info("%s bleu_info: %s", self.count_valid_epoch, bleu_info)
            logger.info("%s mover score: %s", self.count_valid_epoch, moverScore)

            self.count_valid_epoch += 1

        else:
            logger.info('not in')

        return self.check_validation_end(outputs)

    def validation_epoch_end(self, outputs, prefix="val"):
        self.step_count += 1

        if "preds" in outputs[0]:
            output_test_predictions_file = os.path.join(self.hparams.output_dir, "validation_predictions_" +
                                                        str(self.count_valid_epoch) + ".txt")
            output_test_targets_file = os.path.join(self.hparams.output_dir, "validation_targets_" +
                                                        str(self.count_valid_epoch) + ".txt")
            with open(output_test_predictions_file, "w") as p_writer, open(output_test_targets_file, "w") as t_writer:
                for output_batch in outputs:
                    p_writer.writelines(convert_text(s) + "\n" for s in output_batch["preds"])
                    t_writer.writelines(convert_text(s) + "\n" for s in output_batch["target"])

            bleu_info = eval_sacre_bleu(output_test_targets_file, output_test_predictions_file)
            moverScore = eval_mover_score(output_test_targets_file, output_test_predictions_file)

            metrics = {
                f"{prefix}_avg_bleu": bleu_info,
                f"{prefix}_mover_mean1": moverScore[0],
                f"{prefix}_mover_median1": moverScore[1],
                "step_count": self.step_count,
            }

            logger.info("valid epoch: %s", self.count_valid_epoch)
            logger.info("%s bleu_info: %s", self.count_valid_epoch, bleu_info)
            logger.info("%s mover score: %s", self.count_valid_epoch, moverScore)

            avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
            mover_tensor: torch.FloatTensor = torch.tensor(moverScore[0]).type_as(avg_loss)

            self.count_valid_epoch += 1

        else:
            logger.info('not in')
            avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        return {"avg_val_loss": avg_loss, "log": metrics, f"{prefix}_mover": mover_tensor}

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = AgendaDataset(self.tokenizer, type_path=type_path, **self.dataset_kwargs)
        logger.info('loading %s dataloader...', type_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn, shuffle=shuffle,
                                num_workers=4)  # num_workers 값 조정
        logger.info('done')
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("dev", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.test_batch_size)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)
        parser.add_argument(
            "--max_source_length",
            default=250,
            type=int,
            help="The maximum total input sequence length after tokenization.",
        )
        parser.add_argument(
            "--max_target_length",
            default=512,
            type=int,
            help="The maximum total output sequence length after tokenization.",
        )
        parser.add_argument(
            "--data_dir",
            default=None,
            type=str,
            required=True,
            help="The input data dir.",
        )
        parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=-1,
            required=False,
            help="Early stopping patience.",
        )
        parser.add_argument(
            "--checkpoint",
            default=None,
            type=str,
            help="The checkpoint to initialize model.",
        )
        parser.add_argument(
            "--checkpoint_model",
            default=None,
            type=str,
            help="The checkpoint model file.",
        )
        return parser

def main(args):
    if not args.output_dir:
        args.output_dir = os.path.join("./results", f"{args.task}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(args.output_dir)

    model = SummarizationTrainer(args)
    if args.checkpoint_model:
        model = model.load_from_checkpoint(args.checkpoint_model)
        logger.info("args.data_dir: %s", args.data_dir)
        model.dataset_kwargs: dict = dict(
            data_dir=args.data_dir,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
        )
        model.hparams = args

    if args.early_stopping_patience >= 0:
        es_callback = get_early_stopping_callback(model.val_metric, args.early_stopping_patience)
    else:
        es_callback = False

    trainer = generic_train(
        model, args, 
        checkpoint_callback=get_checkpoint_callback(args.output_dir, model.val_metric),
        early_stopping_callback=es_callback
    )

    if args.do_predict:
        if args.checkpoint_model:
            trainer.test(model)
        else:
            checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "*.ckpt"), recursive=True)))
            if checkpoints:
                print('Loading weights from {}'.format(checkpoints[-1]))
                model = model.load_from_checkpoint(checkpoints[-1])
                model.dataset_kwargs: dict = dict(
                    data_dir=args.data_dir,
                    max_source_length=args.max_source_length,
                    max_target_length=args.max_target_length,
                )
                model.hparams = args
            trainer.test(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_args(parser, os.getcwd())
    parser = SummarizationTrainer.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()
    main(args)
