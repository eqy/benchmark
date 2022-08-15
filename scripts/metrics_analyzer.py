import pathlib

from ..components.model_analyzer.tb_dcgm_types.cpu_used_memory import CPUUsedMemory
from ..components.model_analyzer.tb_dcgm_types.gpu_fp32active import GPUFP32Active
from typing import Optional, List, Tuple
from torchbenchmark import ModelTask
import os
import sys
import time
from components.model_analyzer.TorchBenchAnalyzer import ModelAnalyzer
from run_sweep import WORKER_TIMEOUT, WARMUP_ROUNDS, ModelTestResult, NANOSECONDS_PER_MILLISECONDS


def run_one_step_metrics(func, device: str, nwarmup=WARMUP_ROUNDS, num_iter=10) -> Tuple[float, float, Optional[Tuple[torch.Tensor]]]:
    "Run one step of the model, and return the latency in milliseconds."
    # Warm-up `nwarmup` rounds
    for _i in range(nwarmup):
        func()
    result_summary = []
    model_analyzer = ModelAnalyzer()
    model_analyzer.add_metrics([GPUFP32Active, CPUUsedMemory])
    model_analyzer.start_monitor()
    for _i in range(num_iter):
        if device == "cuda":
            torch.cuda.synchronize()
            # Collect time_ns() instead of time() which does not provide better precision than 1
            # second according to https://docs.python.org/3/library/time.html#time.time.
            t0 = time.time_ns()
            func()
            torch.cuda.synchronize()  # Wait for the events to be recorded!
            t1 = time.time_ns()
        else:
            t0 = time.time_ns()
            func()
            t1 = time.time_ns()
        result_summary.append((t1 - t0) / NANOSECONDS_PER_MILLISECONDS)
    model_analyzer.stop_monitor()
    model_analyzer.aggregate()
    tflops = model_analyzer.calculate_flops() 
    wall_latency = numpy.median(result_summary)
    return (wall_latency, tflops)


def _run_model_test_metrics(model_path: pathlib.Path, test: str, device: str, jit: bool, batch_size: Optional[int], extra_args: List[str]) -> ModelTestResult:
    assert test == "train" or test == "eval", f"Test must be either 'train' or 'eval', but get {test}."
    result = ModelTestResult(name=model_path.name, test=test, device=device, extra_args=extra_args, batch_size=None, precision="fp32",
                             status="OK", results={})
    # Run the benchmark test in a separate process
    print(f"Running model {model_path.name} ... ", end='', flush=True)
    status: str = "OK"
    bs_name = "batch_size"
    correctness_name = "correctness"
    error_message: Optional[str] = None
    try:
        task = ModelTask(os.path.basename(model_path), timeout=WORKER_TIMEOUT)
        if not task.model_details.exists:
            status = "NotExist"
            return
        task.make_model_instance(test=test, device=device, jit=jit, batch_size=batch_size, extra_args=extra_args)
        # Check the batch size in the model matches the specified value
        result.batch_size = task.get_model_attribute(bs_name)
        result.precision = task.get_model_attribute("dargs", "precision")
        if batch_size and (not result.batch_size == batch_size):
            raise ValueError(f"User specify batch size {batch_size}, but model {result.name} runs with batch size {result.batch_size}. Please report a bug.")
        result.results["latency_ms"] = run_one_step_metrics(task.invoke, device)
        # if NUM_BATCHES is set, update to per-batch latencies
        num_batches = task.get_model_attribute("NUM_BATCHES")
        if num_batches:
            result.results["latency_ms"] = result.results["latency_ms"] / num_batches
        # if the model provides eager eval result, save it for cosine similarity
        correctness = task.get_model_attribute(correctness_name)
        if correctness is not None:
            result.results[correctness_name] = str(correctness)
    except NotImplementedError as e:
        status = "NotImplemented"
        error_message = str(e)
    except TypeError as e: # TypeError is raised when the model doesn't support variable batch sizes
        status = "TypeError"
        error_message = str(e)
    except KeyboardInterrupt as e:
        status = "UserInterrupted"
        error_message = str(e)
    except Exception as e:
        status = f"{type(e).__name__}"
        error_message = str(e)
    finally:
        print(f"[ {status} ]")
        result.status = status
        if error_message:
            result.results["error_message"] = error_message
        if status == "UserInterrupted":
            sys.exit(1)
        return result