import os
from torch.nn.parallel import DistributedDataParallel as DDP

from torchbenchmark.util.extra_args import is_hf_model


def apply_trainer(model, trainer):
    # supported distributed backends
    if trainer not in ["ddp", "fsdp"]:
        raise RuntimeError(f"Unsupported distributed scheme {trainer} for model")
    if is_hf_model(model):
        # DDP requires to use unwrapped model for huggingface
        module, _inputs = model.get_module(wrap_model=False)
    else:
        module, _inputs = model.get_module()
    if trainer == "ddp":
        if hasattr(model, "enable_ddp") and model.enable_ddp:
            return model.enable_ddp()
        local_rank = int(os.getenv("LOCAL_RANK", -1))
        ddp_module = DDP(
            module,
            device_ids=[local_rank],
            # If buffer broadcast is necessary, specific optimizations might be
            # necessary to optimize performance. Disable it by default.
            broadcast_buffers=False,
            # Set gradient as bucket view to avoid unnecessary copies
            gradient_as_bucket_view=True,
            # TODO: tune bucket_cap_mb
            static_graph=True,
        ) 
        return ddp_module
    elif trainer == "fsdp":
        if hasattr(model, "enable_fsdp") and model.enable_fsdp:
            return model.enable_fsdp()
        import torch
        from torch.distributed.fsdp import FullySharedDataParallel as FSDP
        local_rank = int(os.getenv("LOCAL_RANK", -1))
        torch.cuda.set_device(local_rank)
        fsdp_module = FSDP(
            module,
            device_id = torch.cuda.current_device()
        )
        return fsdp_module
