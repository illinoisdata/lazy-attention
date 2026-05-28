"""
Here we modify the core.py to make it compatible with LazyAttention.
   
We added this is only for:
- Launch LazyEngineCoreProc instead of EngineCoreProc
- The only difference is `MsgpackDecoder(EngineCoreRequest)`

Changed by Haocheng at 2025/09/04
"""

import json
import os
import queue
import signal
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future
from inspect import isclass, signature
from logging import DEBUG
from typing import Any, Callable, Optional, TypeVar, Union

import msgspec
import zmq

from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.executor.multiproc_worker_utils import _add_prefix
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value)
from vllm.utils import resolve_obj_by_qualname, zmq_socket_ctx
from vllm.v1.core.kv_cache_utils import (get_kv_cache_config,
                                         unify_kv_cache_configs)
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler as V1Scheduler
from vllm.v1.engine import (EngineCoreOutputs,
                            EngineCoreRequestType, UtilityOutput)
from vllm.v1.engine.mm_input_cache import MirroredProcessingCache
from vllm.v1.executor.abstract import Executor
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.structured_output import StructuredOutputManager
from vllm.version import __version__ as VLLM_VERSION

from vllm.v1.engine.core import EngineCoreProc, DPEngineCoreProc

from lazy.engine import EngineCoreRequest

logger = init_logger(__name__)

class LazyEngineCoreProc(EngineCoreProc):
    @staticmethod
    def run_engine_core(*args,
                        dp_rank: int = 0,
                        local_dp_rank: int = 0,
                        **kwargs):
        """Launch EngineCore busy loop in background process."""

        # When vLLM forces the `spawn` start method (e.g. once CUDA is
        # initialized in the parent), this child process starts fresh and only
        # imports `lazy.engine.core` to resolve this target function. None of
        # the other monkey-patches (scheduler, model runner, attention backend,
        # rotary embedding) are installed because `lazy.__vllm__` is never
        # imported here. Re-apply them before the engine is constructed so the
        # worker actually runs lazy attention instead of vanilla vLLM.
        from lazy.vllm_patch import apply_all_patches
        apply_all_patches()

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the engine_core
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        engine_core: Optional[EngineCoreProc] = None
        try:
            parallel_config: ParallelConfig = kwargs[
                "vllm_config"].parallel_config
            if parallel_config.data_parallel_size > 1:
                # Set data parallel rank for this engine process.
                parallel_config.data_parallel_rank = dp_rank
                parallel_config.data_parallel_rank_local = local_dp_rank
                engine_core = DPLazyEngineCoreProc(*args, **kwargs)
            else:
                engine_core = LazyEngineCoreProc(*args, **kwargs)
            logger.info(f"Starting LazyEngineCoreProc {engine_core}")
            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception as e:
            if engine_core is None:
                logger.exception("EngineCore failed to start.")
            else:
                logger.exception("EngineCore encountered a fatal error.")
                engine_core._send_engine_dead()
            raise e
        finally:
            if engine_core is not None:
                engine_core.shutdown()
    
    def process_input_socket(self, input_path: str, engine_index: int):
        """Input socket IO thread."""
        logger.info(f"Starting input socket thread for lazy engine {engine_index}")
        # Msgpack serialization decoding.
        add_request_decoder = MsgpackDecoder(EngineCoreRequest)
        generic_decoder = MsgpackDecoder()
        identity = engine_index.to_bytes(length=2, byteorder="little")

        with zmq_socket_ctx(input_path,
                            zmq.DEALER,
                            identity=identity,
                            bind=False) as socket:

            # Send ready message to front-end once input socket is connected.
            socket.send(b'READY')

            while True:
                # (RequestType, RequestData)
                type_frame, *data_frames = socket.recv_multipart(copy=False)
                request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                # Deserialize the request data.
                logger.debug(f"Received request of type {request_type}")
                decoder = add_request_decoder if (
                    request_type
                    == EngineCoreRequestType.ADD) else generic_decoder
                request = decoder.decode(data_frames)
                
                logger.debug(f"Decoded request: {request}")
                # Push to input queue for core busy loop.
                self.input_queue.put_nowait((request_type, request))
                logger.debug(f"Add request {request.request_id} to core busy loop")
    
class DPLazyEngineCoreProc(LazyEngineCoreProc):
    # TODO(haocheng): implement DP version
    pass
    
    
def apply_patch():
    import vllm.v1.engine.core
    vllm.v1.engine.core.EngineCoreProc = LazyEngineCoreProc
