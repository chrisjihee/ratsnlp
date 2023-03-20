import os
from glob import glob
from dataclasses import dataclass, field
from pathlib import Path

from dataclasses_json import DataClassJsonMixin

from chrisbase.io import make_dir


@dataclass
class ClassificationTrainArguments(DataClassJsonMixin):
    pretrained_model_path: str = field(
        default="beomi/kcbert-base",
        metadata={"help": "name/path of the pretrained model"}
    )
    downstream_model_path: str = field(
        default=None,
        metadata={"help": "output model directory path"}
    )
    downstream_model_file: str = field(
        default=None,
        metadata={"help": "output model filename format"}
    )
    downstream_conf_file: str = field(
        default=None,
        metadata={"help": "downstream config filename"}
    )
    downstream_data_home: str = field(
        default="/content/Korpora",
        metadata={"help": "root of the downstream data"}
    )
    downstream_data_name: str = field(
        default=None,
        metadata={"help": "name of the downstream data"}
    )
    downstream_task_name: str = field(
        default="document-classification",
        metadata={"help": "name of the downstream task"}
    )
    max_seq_length: int = field(
        default=128,
        metadata={"help": "The maximum total input sequence length after tokenization. "
                          "Sequences longer than this will be truncated, sequences shorter will be padded."}
    )
    save_top_k: int = field(
        default=1,
        metadata={"help": "save top k model checkpoints"}
    )
    monitor: str = field(
        default="min val_loss",
        metadata={"help": "monitor condition (save top k)"}
    )
    seed: int = field(
        default=None,
        metadata={"help": "random seed"}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "overwrite the cached training and evaluation sets"}
    )
    force_download: bool = field(
        default=False,
        metadata={"help": "force to download downstream data and pretrained models"}
    )
    test_mode: bool = field(
        default=False,
        metadata={"help": "test mode enables `fast_dev_run`"}
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "learning rate"}
    )
    epochs: int = field(
        default=3,
        metadata={"help": "max epochs"}
    )
    batch_size: int = field(
        default=32,
        metadata={"help": "batch size. if 0, let lightening find the best batch size"}
    )
    cpu_workers: int = field(
        default=os.cpu_count(),
        metadata={"help": "number of CPU workers"}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "enable train on floating point 16"}
    )

    def save_config(self) -> Path:
        config_file = make_dir(self.downstream_model_path) / self.downstream_conf_file
        config_file.write_text(self.to_json(ensure_ascii=False, indent=2, default=str))
        return config_file


@dataclass
class ClassificationDeployArguments:

    def __init__(
            self,
            pretrained_model_name=None,
            downstream_model_dir=None,
            downstream_model_checkpoint_fpath=None,
            max_seq_length=128,
    ):
        self.pretrained_model_name = pretrained_model_name
        self.max_seq_length = max_seq_length
        if downstream_model_checkpoint_fpath is not None:
            self.downstream_model_checkpoint_fpath = downstream_model_checkpoint_fpath
        elif downstream_model_dir is not None:
            ckpt_file_names = glob(os.path.join(downstream_model_dir, "*.ckpt"))
            ckpt_file_names = [el for el in ckpt_file_names if "temp" not in el and "tmp" not in el]
            if len(ckpt_file_names) == 0:
                raise Exception(f"downstream_model_dir \"{downstream_model_dir}\" is not valid")
            selected_fname = ckpt_file_names[-1]
            min_val_loss = os.path.split(selected_fname)[-1].replace(".ckpt", "").split("=")[-1].split("-")[0]
            try:
                for ckpt_file_name in ckpt_file_names:
                    val_loss = os.path.split(ckpt_file_name)[-1].replace(".ckpt", "").split("=")[-1].split("-")[0]
                    if float(val_loss) < float(min_val_loss):
                        selected_fname = ckpt_file_name
                        min_val_loss = val_loss
            except:
                raise Exception(f"the ckpt file name of downstream_model_directory \"{downstream_model_dir}\" is not valid")
            self.downstream_model_checkpoint_fpath = selected_fname
        else:
            raise Exception("Either downstream_model_dir or downstream_model_checkpoint_fpath must be entered.")
        print(f"downstream_model_checkpoint_fpath: {self.downstream_model_checkpoint_fpath}")
