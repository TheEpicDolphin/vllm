# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.config import VllmConfig


def init_speculator(
    vllm_config: VllmConfig,
    device: torch.device,
    ssd_comm=None,
    is_ssd_speculator: bool = False,
):
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None

    if speculative_config.enable_ssd:
        assert ssd_comm is not None
        from vllm.v1.worker.gpu.spec_decode.ssd.speculator import SSDSpeculator

        return SSDSpeculator(
            vllm_config,
            device,
            is_speculator=is_ssd_speculator,
            ssd_comm=ssd_comm,
        )

    if speculative_config.use_eagle():
        from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator

        return EagleSpeculator(vllm_config, device)
    raise NotImplementedError(f"{speculative_config.method} is not supported yet.")
