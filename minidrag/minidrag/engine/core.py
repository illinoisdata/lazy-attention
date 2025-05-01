"""here we modify the core.py to make it compatible with the dynamic rag EngineCoreRequest"""

import os
import queue
import signal
import sys
import threading
import time
from concurrent.futures import Future
from inspect import isclass, signature
from logging import DEBUG
from typing import Any, Callable, Optional, TypeVar, Union

import msgspec
import psutil
import zmq
import zmq.asyncio

from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.executor.multiproc_worker_utils import _add_prefix
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value)
from vllm.utils import (get_exception_traceback, resolve_obj_by_qualname,
                        zmq_socket_ctx)
from vllm.v1.core.kv_cache_utils import (get_kv_cache_config,
                                         unify_kv_cache_configs)
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler as V1Scheduler
from vllm.v1.engine import (EngineCoreOutputs, EngineCoreRequest,
                            EngineCoreRequestType, UtilityOutput)
from vllm.v1.engine.mm_input_cache import MMInputCacheServer
from vllm.v1.executor.abstract import Executor
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.structured_output import StructuredOutputManager
from vllm.version import __version__ as VLLM_VERSION

from minidrag.engine import EngineCoreRequest


# class EngineCoreProc(EngineCore):
def process_input_socket(self, input_path: str):
    """Input socket IO thread."""
    # Msgpack serialization decoding.
    add_request_decoder = MsgpackDecoder(EngineCoreRequest)
    generic_decoder = MsgpackDecoder()
    # add_request_decoder = MsgpackDecoder()
    with zmq_socket_ctx(input_path, zmq.constants.PULL) as socket:
        while True:
            # (RequestType, RequestData)
            type_frame, data_frame = socket.recv_multipart(copy=False)
            request_type = EngineCoreRequestType(bytes(type_frame.buffer))
            # Deserialize the request data.
            decoder = add_request_decoder if (
                request_type
                == EngineCoreRequestType.ADD) else generic_decoder
            request = decoder.decode(data_frame.buffer)
            # print(f"Recived: {data_frame.buffer.hex()}")
            # print("decoded request after patch:", request)
            # Push to input queue for core busy loop.
            self.input_queue.put_nowait((request_type, request))
            
            

def apply_patch():
    import vllm.v1.engine.core
    vllm.v1.engine.core.EngineCoreProc.process_input_socket = process_input_socket