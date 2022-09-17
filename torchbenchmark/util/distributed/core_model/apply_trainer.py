import os
from torch.nn.parallel import DistributedDataParallel as DDP

from torchbenchmark.util.extra_args import is_hf_model


def apply_trainer(model, trainer):
    # supported distributed backends
    if trainer not in ["ddp", "fsdp"]:
        raise RuntimeError(f"Unsupported distributed scheme {trainer} for model")
    if trainer == "ddp":
        local_rank = int(os.getenv("LOCAL_RANK", -1))
        ddp_model = DDP(
            model,
            device_ids=[local_rank],
            # If buffer broadcast is necessary, specific optimizations might be
            # necessary to optimize performance. Disable it by default.
            broadcast_buffers=False,
            # Set gradient as bucket view to avoid unnecessary copies
            gradient_as_bucket_view=True,
            # TODO: tune bucket_cap_mb
            static_graph=True,
        ) 
        return ddp_model
    elif trainer == "fsdp":
        assert is_hf_model(model)
        from torch.distributed.fsdp import FullySharedDataParallel as FSDP