# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2023 The LLM-on-Ray Authors.

#!/usr/bin/env python

import os
import torch
from typing import Any, Dict

import ray
import transformers
from peft import LoraConfig, get_peft_model
from ray.air import FailureConfig, RunConfig
from ray.air.config import ScalingConfig
from ray.train.torch import TorchTrainer

from comps import CustomLogger
from comps.finetuning.llm_on_ray import common
from comps.finetuning.llm_on_ray.finetune.finetune import (
    adapt_transformers_to_device,
    set_seed,
    convert_dtype,
    load_tokenizer,
    load_dataset,
    tokenize_dataset,
    prepare_data_collator,
    load_model,
    get_trainer,
    get_finetune_config
)

original_load_model_func = load_model
logger = CustomLogger("llm_on_ray/sqft")


def load_model(config: Dict):
    if config["General"].get("lora_config", None) and config["General"].get("task", "instruction_tuning") == "instruction_tuning":
        model_name = config["General"]["base_model"]
        model_dtype = convert_dtype(config["Training"].get("mixed_precision", "no"))
        model_config = config["General"].get("config", {})
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=model_dtype, **model_config)
        lora_config = config["General"].get("lora_config", None)
        neural_lora_search = lora_config.pop("neural_lora_search", False)
        nls_target_modules = lora_config.pop("nls_target_modules", None)
        search_space = lora_config.pop("search_space", None)
        peft_config = LoraConfig(**lora_config)
        model = get_peft_model(model, peft_config)

        if neural_lora_search:
            from modules.elastic_lora_linear import make_lora_elastic
            make_lora_elastic(
                model, 
                search_space, 
                nls_target_modules, 
                config_save_dir=config["General"]["output_dir"], 
            )
        egc = config["General"].get("enable_gradient_checkpointing", False)
        if egc:
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable()
            model.config.use_cache = False

        model.to(dtype=model_dtype, device=torch.device(config["Training"]["device"]))

        return model, None
    else:
        return original_load_model_func(config)


# The following code is copied from
# https://github.com/opea-project/GenAIComps/blob/v1.1/comps/finetuning/llm_on_ray/finetune/finetune.py#L475-L505
def train_func(config: Dict[str, Any]):
    os.chdir(config["cwd"])

    adapt_transformers_to_device(config)

    set_seed(config)

    tokenizer = load_tokenizer(config)

    dataset = load_dataset(config)

    max_train_samples = config["Dataset"].get("max_train_samples", 0)
    if 0 < max_train_samples < len(dataset["train"]):
        dataset["train"] = dataset["train"].select(range(max_train_samples))

    max_eval_samples = config["Dataset"].get("max_eval_samples", 0)
    if "validation" in dataset and 0 < max_eval_samples < len(dataset["validation"]):
        dataset["validation"] = dataset["validation"].select(range(max_eval_samples))

    tokenized_dataset = tokenize_dataset(config, tokenizer, dataset)

    data_collator = prepare_data_collator(config, tokenizer)

    model, ref_model = load_model(config)

    training_args, trainer = get_trainer(config, model, ref_model, tokenizer, tokenized_dataset, data_collator)

    logger.info("train start")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    logger.info("train finish")


# The following code is copied from
# https://github.com/jpablomch/GenAIResearch/blob/main/comps/sqft/llm_on_ray/finetune/finetune.py#L105-L197
def main(external_config=None):
    if not external_config:
        config = get_finetune_config()
    else:
        config = external_config

    config["cwd"] = os.getcwd()

    num_training_workers = config["Training"].get("num_training_workers")
    resources_per_worker = config["Training"].get("resources_per_worker")

    if num_training_workers > 1 and config["Training"].get("accelerate_mode", None) is None:
        config["Training"]["accelerate_mode"] = "DDP"  # will use DDP to accelerate if no method specified

    ccl_worker_count = 1
    device = config["Training"]["device"]
    if device != "cpu":
        ccl_worker_count = num_training_workers

    if not ray.is_initialized():
        runtime_env = {
            "env_vars": {
                "OMP_NUM_THREADS": str(resources_per_worker["CPU"]),
                "CCL_ZE_IPC_EXCHANGE": "sockets",
                "CCL_WORKER_COUNT": str(ccl_worker_count),
                "CCL_LOG_LEVEL": "info",
                "FI_TCP_IFACE": "lo",
                "FI_PROVIDER": "tcp",
            }
        }

        if config["General"]["gpt_base_model"] is True:
            runtime_env["pip"] = ["transformers==4.26.0"]

        if device == "gpu":
            num_cpus = resources_per_worker["CPU"] * num_training_workers + 1  # additional 1 for head worker
            ray.init(num_cpus=num_cpus, runtime_env=runtime_env)
        else:
            ray.init(runtime_env=runtime_env)

    logger.info(f"ray available resources = {ray.available_resources()}")

    use_gpu = True if device == "cuda" else False
    scaling_config = ScalingConfig(
        num_workers=num_training_workers,
        use_gpu=use_gpu,
        resources_per_worker=resources_per_worker,
        placement_strategy="SPREAD",
    )

    # if try to use Intel GPU, convert device to 'xpu'
    # due to accelerate internal use 'xpu' represent Intel GPU
    if device == "gpu":
        from accelerate.utils import is_xpu_available

        if is_xpu_available():
            device = "xpu"

    if config.get("torch_config", None) is None:
        backend = None
        if device == "cpu" or device == "xpu" or device == "gpu":
            backend = "ccl"
        elif device == "hpu":
            backend = "hccl"
        torch_config = common.TorchConfig(backend=backend, device=device)
    else:
        customer_torch_config = config.get("torch_config")
        torch_config = common.TorchConfig(**customer_torch_config, device=device)

    if config.get("failure_config", None) is None:
        failure_config = FailureConfig()
    else:
        customer_failure_config = config.get("failure_config")
        failure_config = FailureConfig(**customer_failure_config)

    if config.get("run_config", None) is None:
        run_config = RunConfig(failure_config=failure_config)
    else:
        customer_run_config = config.get("run_config")
        if customer_run_config.get("failure_config", None) is None:
            customer_run_config["failure_config"] = failure_config
        run_config = RunConfig(**customer_run_config)

    trainer = TorchTrainer(
        train_func,
        train_loop_config=config,
        scaling_config=scaling_config,
        torch_config=torch_config,
        run_config=run_config,
    )
    results = trainer.fit()
    if external_config is not None:
        return results


if __name__ == "__main__":
    main()
