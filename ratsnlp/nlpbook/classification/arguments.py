import os
from dataclasses import dataclass, field
from pathlib import Path

from dataclasses_json import DataClassJsonMixin

from chrisbase.io import make_dir, files


@dataclass
class ClassificationTrainArguments(DataClassJsonMixin):
    working_config_file: str | None = field(
        default=None,
        metadata={"help": "downstream config filename"}
    )
    pretrained_model_path: str | None = field(
        default="beomi/kcbert-base",
        metadata={"help": "name/path of the pretrained model"}
    )
    downstream_model_path: str | None = field(
        default=None,
        metadata={"help": "output model directory path"}
    )
    downstream_model_file: str | None = field(
        default=None,
        metadata={"help": "output model filename format"}
    )
    downstream_data_home: str | None = field(
        default="/content/Korpora",
        metadata={"help": "root of the downstream data"}
    )
    downstream_data_name: str | None = field(
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
    seed: int | None = field(
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

    def save_working_config(self) -> Path:
        config_file = make_dir(self.downstream_model_path) / self.working_config_file
        config_file.write_text(self.to_json(ensure_ascii=False, indent=2, default=str))
        return config_file


@dataclass
class ClassificationDeployArguments(DataClassJsonMixin):
    working_config_file: str | None = field(
        default=None,
        metadata={"help": "downstream config filename"}
    )
    pretrained_model_path: str | None = field(
        default="beomi/kcbert-base",
        metadata={"help": "name/path of the pretrained model"}
    )
    downstream_model_path: str | None = field(
        default=None,
        metadata={"help": "output model directory path"}
    )
    downstream_model_file: str | None = field(
        default=None,
        metadata={"help": "output model filename"}
    )
    max_seq_length: int = field(
        default=128,
        metadata={"help": "The maximum total input sequence length after tokenization. "
                          "Sequences longer than this will be truncated, sequences shorter will be padded."}
    )

    def __post_init__(self):
        if self.downstream_model_file is None:
            ckpt_files = files(Path(self.downstream_model_path) / "*.ckpt")
            ckpt_files = sorted([x for x in ckpt_files if "temp" not in str(x) and "tmp" not in str(x)], key=str)
            assert len(ckpt_files) > 0, f"No checkpoint file in {self.downstream_model_path}"
            self.downstream_model_file = ckpt_files[-1].name
            print(f"downstream_model_file: {self.downstream_model_file}")

    def save_working_config(self) -> Path:
        config_file = make_dir(self.downstream_model_path) / self.working_config_file
        config_file.write_text(self.to_json(ensure_ascii=False, indent=2, default=str))
        return config_file
