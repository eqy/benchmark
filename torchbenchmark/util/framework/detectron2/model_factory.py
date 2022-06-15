from torchbenchmark.util.model import BenchmarkModel
import itertools
import os
from pathlib import Path
import torch

# setup environment variable
CURRENT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(CURRENT_DIR.parent.parent.parent, "data", ".data", "coco2017-minimal")
assert os.path.exists(DATA_DIR), "Couldn't find coco2017 minimal data dir, please run install.py again."
if not 'DETECTRON2_DATASETS' in os.environ:
    os.environ['DETECTRON2_DATASETS'] = DATA_DIR

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.engine import default_argument_parser
from detectron2.solver import build_optimizer
from detectron2.config import LazyConfig, get_cfg, instantiate
from detectron2 import model_zoo
from detectron2.utils.events import EventStorage
from torch.utils._pytree import tree_map
from detectron2.checkpoint import DetectionCheckpointer

from typing import Tuple

def setup(args):
    if args.config_file.endswith(".yaml"):
        cfg = get_cfg()
        cfg.merge_from_file(args.config_file)
        cfg.SOLVER.BASE_LR = 0.001  # Avoid NaNs. Not useful in this script anyway.
        # set images per batch to 1
        cfg.SOLVER.IMS_PER_BATCH = 1
        cfg.MODEL.WEIGHTS = args.model_file
        cfg.merge_from_list(args.opts)
        cfg.freeze()
    else:
        cfg = LazyConfig.load(args.config_file)
        cfg = LazyConfig.apply_overrides(cfg, args.opts)
    return cfg

def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    return model

def prefetch(dataloader, device, precision="fp32"):
    r = []
    dtype = torch.float16 if precision == "fp16" else torch.float32
    for batch in dataloader:
        r.append(tree_map(lambda x: x.to(device, dtype=dtype) if isinstance(x, torch.Tensor) else x, batch))
    return r

def get_abs_path(config):
    import detectron2
    detectron2_root = os.path.abspath(os.path.dirname(detectron2.__file__))
    return os.path.join(detectron2_root, "model_zoo", "configs", config)

class Detectron2Model(BenchmarkModel):
    # To recognize this is a detectron2 model
    DETECTRON2_MODEL = True
    # Default eval precision on CUDA device is fp16
    DEFAULT_EVAL_CUDA_PRECISION = "fp16"
    # Default batch sizes
    DEFAULT_TRAIN_BSIZE = 1
    DEFAULT_EVAL_BSIZE = 1

    def __init__(self, variant, test, device, jit=False, batch_size=None, extra_args=[]):
        super().__init__(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        # load model file
        assert hasattr(self, "model_file"), f"Detectron2 models must specify its model_file."
        assert (os.path.exists(self.model_file)), f"Detectron2 model file specified {self.model_file} doesn't exist."
        parser = default_argument_parser()
        args = parser.parse_args(["--config-file", get_abs_path(variant)])
        # setup pre-trained model weights
        args.model_file = self.model_file
        data_cfg = model_zoo.get_config("common/data/coco.py").dataloader
        cfg = setup(args)
        self.model = build_model(cfg).to(self.device)
        if self.test == "train":
            self.model.train()
            self.optimizer = build_optimizer(cfg, self.model)
            # setup train dataset
            data_cfg.train.dataset.names = "coco_2017_val_100"
            data_cfg.train.total_batch_size = self.batch_size
            train_loader = instantiate(data_cfg.train)
            self.example_inputs = prefetch(itertools.islice(train_loader, 100), self.device)
        elif self.test == "eval":
            # load model from pretrained checkpoint
            DetectionCheckpointer(self.model).load(self.model_file)
            self.model.eval()
            # setup eval dataset
            data_cfg.test.dataset.names = "coco_2017_val_100"
            data_cfg.test.batch_size = self.batch_size
            test_loader = instantiate(data_cfg.test)
            self.example_inputs = prefetch(itertools.islice(test_loader, 100), self.device)
        self.NUM_BATCHES = len(self.example_inputs)
        cfg.defrost()

    def get_module(self):
        return self.model, self.example_inputs

    def enable_fp16_half(self):
        assert self.dargs.precision == "fp16", f"Expected precision fp16, get {self.dargs.precision}"
        self.model = self.model.half()
        self.example_inputs = prefetch(self.example_inputs, self.device, self.dargs.precision)

    def train(self):
        with EventStorage():
            for batch_id in range(self.NUM_BATCHES):
                loss_dict = self.model(self.example_inputs[batch_id])
                if isinstance(loss_dict, torch.Tensor):
                    losses = loss_dict
                    loss_dict = {"total_loss": loss_dict}
                else:
                    losses = sum(loss_dict.values())
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

    def eval(self) -> Tuple[torch.Tensor]:
        with torch.no_grad():
            for batch_id in range(self.NUM_BATCHES):
                out = self.model(self.example_inputs[batch_id])
        # retrieve output tensors
        outputs = []
        for item in out:
            fields = list(map(lambda x: list(x.get_fields().values()), item.values()))
            for boxes in fields:
                tensor_box = list(filter(lambda x: isinstance(x, torch.Tensor), boxes))
                outputs.extend(tensor_box)
        return tuple(outputs)