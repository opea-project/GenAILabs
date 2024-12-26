# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from fastapi import BackgroundTasks

from comps import opea_microservices, register_microservice
from comps.finetuning.finetuning_service import (
    create_finetuning_jobs,
    list_finetuning_jobs,
    retrieve_finetuning_job,
    cancel_finetuning_job,
    upload_training_files,
    list_checkpoints,
)

from sqft_config import ExtractSubAdapterParams, FineTuningParams, MergeAdapterParams
from handlers import handle_create_sqft_jobs, handle_extract_sub_adapter, handle_merge_adapter


@register_microservice(name="opea_service@finetuning", endpoint="/v1/sqft/jobs", host="0.0.0.0", port=8015)
def create_sqft_jobs(request: FineTuningParams, background_tasks: BackgroundTasks):
    return handle_create_sqft_jobs(request, background_tasks)


@register_microservice(
    name="opea_service@finetuning", endpoint="/v1/sqft/extract_sub_adapter", host="0.0.0.0", port=8015
)
def extract_sub_adapter(request: ExtractSubAdapterParams):
    return handle_extract_sub_adapter(request)


@register_microservice(name="opea_service@finetuning", endpoint="/v1/sqft/merge_adapter", host="0.0.0.0", port=8015)
def merge_adapter(request: MergeAdapterParams):
    return handle_merge_adapter(request)


if __name__ == "__main__":
    opea_microservices["opea_service@finetuning"].start()
