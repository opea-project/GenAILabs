# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import random
import time
import uuid

from fastapi import BackgroundTasks, HTTPException
from pydantic_yaml import parse_yaml_file_as, to_yaml_file
from ray.job_submission import JobSubmissionClient

from comps import CustomLogger
from comps.cores.proto.api_protocol import FineTuningJob
from comps.finetuning import handlers
from comps.finetuning.handlers import (
    DATASET_BASE_PATH,
    JOBS_PATH,
    OUTPUT_DIR,
    update_job_status,
)

from sqft_config import ExtractSubAdapterParams, FinetuneConfig, FineTuningParams, MergeAdapterParams

logger = CustomLogger("sqft_handlers")


def handle_create_sqft_jobs(request: FineTuningParams, background_tasks: BackgroundTasks):
    base_model = request.model
    train_file = request.training_file
    train_file_path = os.path.join(DATASET_BASE_PATH, train_file)

    if not os.path.exists(train_file_path):
        raise HTTPException(status_code=404, detail=f"Training file '{train_file}' not found!")

    finetune_config = FinetuneConfig(General=request.General, Dataset=request.Dataset, Training=request.Training)
    finetune_config.General.base_model = base_model
    finetune_config.Dataset.train_file = train_file_path
    if request.hyperparameters is not None:
        if request.hyperparameters.epochs != "auto":
            finetune_config.Training.epochs = request.hyperparameters.epochs

        if request.hyperparameters.batch_size != "auto":
            finetune_config.Training.batch_size = request.hyperparameters.batch_size

        if request.hyperparameters.learning_rate_multiplier != "auto":
            finetune_config.Training.learning_rate = request.hyperparameters.learning_rate_multiplier

    if os.getenv("HF_TOKEN", None):
        finetune_config.General.config.token = os.getenv("HF_TOKEN", None)

    job = FineTuningJob(
        id=f"ft-job-{uuid.uuid4()}",
        model=base_model,
        created_at=int(time.time()),
        training_file=train_file,
        hyperparameters={
            "n_epochs": finetune_config.Training.epochs,
            "batch_size": finetune_config.Training.batch_size,
            "learning_rate_multiplier": finetune_config.Training.learning_rate,
        },
        status="running",
        seed=random.randint(0, 1000) if request.seed is None else request.seed,
    )
    finetune_config.General.output_dir = os.path.join(OUTPUT_DIR, job.id)
    if os.getenv("DEVICE", ""):
        logger.info(f"specific device: {os.getenv('DEVICE')}")

        finetune_config.Training.device = os.getenv("DEVICE")
        if finetune_config.Training.device == "hpu":
            if finetune_config.Training.resources_per_worker.HPU == 0:
                # set 1
                finetune_config.Training.resources_per_worker.HPU = 1

    finetune_config_file = f"{JOBS_PATH}/{job.id}.yaml"
    to_yaml_file(finetune_config_file, finetune_config)

    handlers.ray_client = JobSubmissionClient() if handlers.ray_client is None else handlers.ray_client

    ray_job_id = handlers.ray_client.submit_job(
        # Entrypoint shell command to execute
        entrypoint=f"python sqft_runner.py --config_file {finetune_config_file}",
    )

    logger.info(f"Submitted Ray job: {ray_job_id} ...")

    handlers.running_finetuning_jobs[job.id] = job
    handlers.finetuning_job_to_ray_job[job.id] = ray_job_id

    background_tasks.add_task(update_job_status, job.id)

    return job


def handle_extract_sub_adapter(request: ExtractSubAdapterParams):
    fine_tuning_job_id = request.fine_tuning_job_id
    finetune_config_file = f"{JOBS_PATH}/{fine_tuning_job_id}.yaml"
    finetune_config = parse_yaml_file_as(FinetuneConfig, finetune_config_file)

    job = handlers.running_finetuning_jobs.get(fine_tuning_job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job '{fine_tuning_job_id}' not found!")
    finetuned_model_path = os.path.join(OUTPUT_DIR, fine_tuning_job_id)
    assert finetuned_model_path == finetune_config.General.output_dir
    if not os.path.exists(finetuned_model_path):
        raise HTTPException(
            status_code=404,
            detail=f"The fine-tuned model saved by the fine-tuning job '{fine_tuning_job_id}' was not found!",
        )
    if job.status != "succeeded":
        raise HTTPException(status_code=404, detail=f"Fine-tuning job '{fine_tuning_job_id}' has not completed!")

    if finetune_config.General.lora_config is None:
        raise HTTPException(
            status_code=404,
            detail=f"The fine-tuning job '{fine_tuning_job_id}' does not enable LoRA adapter fine-tuning!",
        )
    if not finetune_config.General.lora_config.neural_lora_search:
        raise HTTPException(
            status_code=404,
            detail=f"The fine-tuning job '{fine_tuning_job_id}' did not enable NLS algorithm, "
            f"there is no need to extract sub-adapters!",
        )

    from utils.extract_sub_adapter import main as extract_sub_adapter_main

    extract_sub_adapter_main(
        adapter_model_path=finetuned_model_path,
        adapter_version=request.adapter_version,
        custom_config=request.custom_config,
    )

    return fine_tuning_job_id


def handle_merge_adapter(request: MergeAdapterParams):
    fine_tuning_job_id = request.fine_tuning_job_id
    finetune_config_file = f"{JOBS_PATH}/{fine_tuning_job_id}.yaml"
    finetune_config = parse_yaml_file_as(FinetuneConfig, finetune_config_file)

    job = handlers.running_finetuning_jobs.get(fine_tuning_job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job '{fine_tuning_job_id}' not found!")
    finetuned_model_path = os.path.join(OUTPUT_DIR, fine_tuning_job_id)
    assert finetuned_model_path == finetune_config.General.output_dir
    if not os.path.exists(finetuned_model_path):
        raise HTTPException(
            status_code=404,
            detail=f"The fine-tuned model saved by the fine-tuning job '{fine_tuning_job_id}' was not found!",
        )
    if job.status != "succeeded":
        raise HTTPException(status_code=404, detail=f"Fine-tuning job '{fine_tuning_job_id}' has not completed!")

    if finetune_config.General.lora_config is None:
        raise HTTPException(
            status_code=404,
            detail=f"The fine-tuning job '{fine_tuning_job_id}' does not enable LoRA adapter fine-tuning!",
        )

    adapter_path = finetuned_model_path
    adapter_version = request.adapter_version
    if adapter_version is not None:
        adapter_path = os.path.join(adapter_path, adapter_version)
        if not os.path.exists(adapter_path):
            raise HTTPException(
                status_code=404,
                detail=f"The fine-tuning job '{fine_tuning_job_id}' does not have a '{adapter_version}' adapter!",
            )

    from utils.merge_adapter import main as merge_adapter_main

    merge_adapter_main(
        base_model_path=finetune_config.General.base_model,
        adapter_model_path=adapter_path,
        output_path=os.path.join(adapter_path, "merged_model"),
    )

    return fine_tuning_job_id
