# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2023 The LLM-on-Ray Authors.

from typing import List, Optional, Union

from comps.finetuning.finetune_config import (
    GeneralConfig as BaseGeneralConfig,
    FinetuneConfig as BaseFinetuneConfig,
    FineTuningParams as BaseFineTuningParams,
    LoraConfig,
)

from comps.cores.proto.api_protocol import FineTuningJobIDRequest


class SQFTConfig(LoraConfig):
    neural_lora_search: bool = False
    nls_target_modules: Optional[List[str]] = None
    search_space: Optional[List[int]] = None
    sparse_adapter: bool = False


class GeneralConfig(BaseGeneralConfig):
    lora_config: Optional[Union[LoraConfig, SQFTConfig]] = LoraConfig()


class FinetuneConfig(BaseFinetuneConfig):
    General: GeneralConfig = GeneralConfig()


class FineTuningParams(BaseFineTuningParams):
    General: GeneralConfig = GeneralConfig()


class ExtractSubAdapterParams(FineTuningJobIDRequest):
    adapter_version: str = "heuristic"
    custom_config: Optional[List[int]] = None


class MergeAdapterParams(FineTuningJobIDRequest):
    adapter_version: Optional[str] = None
