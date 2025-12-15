# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.model_executor import set_random_seed
from vllm.platforms import current_platform
from vllm_gaudi.v1.worker.hpu_worker import HPUWorker, init_worker_distributed_environment

from vllm_omni.worker.hpu.hpu_generation_model_runner import HPUGenerationModelRunner


class HPUGenerationWorker(HPUWorker):
    """HPU generation worker for code2wav stage in Omni model."""

    def init_device(self) -> None:
        self.device = torch.device("hpu")
        current_platform.set_device(self.device)
        init_worker_distributed_environment(self.vllm_config, self.rank, self.distributed_init_method, self.local_rank)
        set_random_seed(self.model_config.seed)
        self.model_runner: HPUGenerationModelRunner = HPUGenerationModelRunner(self.vllm_config, self.device)
        self.init_profiler()
