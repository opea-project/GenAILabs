# SQFT Microservice

This repository introduces the microservice for an innovative method **SQFT**:
- **Paper**: [SQFT: Low-cost Model Adaptation in Low-precision Sparse Foundation Models](https://arxiv.org/abs/2410.03750)
- **Official implementation**: [https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning/tree/main/SQFT](https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning/tree/main/SQFT)

SQFT is an end-to-end solution for low-precision sparse parameter-efficient fine-tuning of LLMs. It allows for effective model manipulation in resource-constrained environments.
Specifically, the highlights of SQFT include:

- **SparsePEFT**, an efficient and effective strategy for fine-tuning sparse models. It ensures the preservation of the base model's sparsity during merging through the use of sparse adapters.
- Introduction of quantization scenarios (sparse and quantization). **QA-SparsePEFT** built on SparsePEFT, which allows PEFT fine-tuning to achieve a single INT4 and sparse model adapted to the specific domain (pending support).
- Adopt **Neural Low-rank Adapter Search (NLS)** strategy into all pipelines and solutions. 

Please refer to the [paper](https://arxiv.org/abs/2410.03750) and official [code](https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning/tree/main/SQFT) for more details.

## 🚀 1. Start SQFT Microservice with Python (Option 1)

### 1.1 Install Requirements

```bash

SQFT_path=$PWD
git clone https://github.com/opea-project/GenAIComps
cd GenAIComps
git checkout v1.1
pip install -e .
pushd ${SQFT_path}

# same as https://github.com/opea-project/GenAIComps/tree/main/comps/finetuning#11-install-requirements
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install intel-extension-for-pytorch
python -m pip install oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/
pip install -r GenAIComps/comps/finetuning/requirements.txt

# Install the peft library and apply modifications to support the SparsePEFT strategy in SQFT
git clone https://github.com/huggingface/peft.git
pushd peft
git checkout v0.10.0 
git apply --ignore-space-change --ignore-whitespace ${SQFT_path}/patches/peft-v0.10.0.patch 
pip install -e . 
```

### 1.2 Start SQFT Service with Python Script

#### 1.2.1 Start Ray Cluster

OneCCL and Intel MPI libraries should be dynamically linked in every node before Ray starts:

```bash
source $(python -c "import oneccl_bindings_for_pytorch as torch_ccl; print(torch_ccl.cwd)")/env/setvars.sh
```

Start Ray locally using the following command.

```bash
ray start --head
```

For a multi-node cluster, start additional Ray worker nodes with below command.

```bash
ray start --address='${head_node_ip}:6379'
```

#### 1.2.2 Start SQFT Service

```bash
export HF_TOKEN=${your_huggingface_token}
python sqft_service.py
```

## 🚀2. Start SQFT Microservice with Docker (Option 2)

### 2.1 Setup on CPU

#### 2.1.1 Build Docker Image

Build docker image with below command:

```bash
export HF_TOKEN=${your_huggingface_token}
cd ../../
docker build -t opea/sqft:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy --build-arg HF_TOKEN=$HF_TOKEN -f comps/sqft/Dockerfile .
```

#### 2.1.2 Run Docker with CLI

Start docker container with below command:

```bash
docker run -d --name="sqft-server" -p 8015:8015 --runtime=runc --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy opea/sqft:latest
```

## 🚀 3. Consume the SQFT Service

We use [Arc-E](https://huggingface.co/datasets/allenai/ai2_arc/viewer/ARC-Easy) dataset as a simple example to demonstrate how to use SQFT on OPEA Microservice.

### 3.1 Prepare and upload a training file

First, we need to process the training dataset into the instruction format, for example:

```json
{
    "instruction": "Which factor will most likely cause a person to develop a fever?",
    "input": "",
    "output": "a bacterial population in the bloodstream"
}
```
Here, we use the Arc-E dataset as an example. The processing of the Arc-E training set is performed via the script [example_dataset/preprocess_arc.py](./example_dataset/preprocess_arc.py). 
After obtaining the processed dataset file [arce_train_instruct.json](./example_dataset/arce_train_instruct.json), we can upload it to the server with this command:
```bash
# upload a training file
curl http://<your ip>:8015/v1/files -X POST -H "Content-Type: multipart/form-data" -F "file=@example_dataset/arce_train_instruct.json" -F purpose="fine-tune"
```

### 3.2 Create a fine-tuning job

After uploading a training file, use the following commands to launch some fine-tuning jobs.

#### 3.2.1 Neural LoRA Search (NLS)

Here is an example of using the [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) model with NLS fine-tuning. The LoRA target modules are `q_proj`, `k_proj` and `v_proj`, which are elastic, and the low-rank search space is `[16, 12, 8]`. The result of this training is to obtain a trained super-adapter.

```bash
# Max LoRA rank: 16
#   LoRA target modules            -> Low-rank search space
#   ["q_proj", "k_proj", "v_proj"] -> [16,12,8]
curl http://<your ip>:8015/v1/sqft/jobs \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "training_file": "arce_train_instruct.json",
    "model": "meta-llama/Llama-3.2-1B",
    "General": {
      "lora_config": {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj"],
        "neural_lora_search": true,
        "nls_target_modules": ["q_proj", "k_proj", "v_proj"],
        "search_space": [16, 12, 8]
      }
    },
    "Training": {
      "learning_rate": 1e-04, "epochs": 5, "batch_size": 16
    },
    "Dataset": {
      "max_length": 256
    }
  }'
```

Below are some explanations for the parameters related to the NLS algorithm:

- `neural_lora_search` indicates whether the Neural LoRA Search (NLS) algorithm is enabled.
- `nls_target_modules` specifies the target modules for the NLS strategy, indicating which adapters need to become elastic.
- `search_space` specifies the search space for each target module (adapter). Here, we use `[16, 12, 8]`, meaning that the possible rank for each adapter is [16, 12, 8].

#### 3.2.2 SparsePEFT

SparsePEFT is designed for the foundation model that has been sparsified using any sparse algorithms.
For sparse model selection, SQFT offers several sparse and quantized base models for users to choose from (the complete SQFT scheme). 
Please refer to [here](https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning/tree/main/SQFT#released-foundation-models-) or the [HuggingFace SQFT Model Collection](https://huggingface.co/collections/IntelLabs/sqft-66cd56f90b240963f9cf1a67).
Here is an example of enabling SparsePEFT by setting `sparse_adapter` to True, allowing the adapter to be integrated into the base model without losing sparsity.

```bash
# Max LoRA rank: 16
#   LoRA target modules            -> Low-rank search space
#   ["q_proj", "k_proj", "v_proj"] -> [16,12,8]
curl http://<your ip>:8015/v1/sqft/jobs \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "training_file": "arce_train_instruct.json",
    "model": <path to sparse model>,
    "General": {
      "lora_config": {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj"],
        "sparse_adapter": true
      }
    },
    "Training": {
      "learning_rate": 1e-04, "epochs": 5, "batch_size": 16
    },
    "Dataset": {
      "max_length": 256
    }
  }'
```

Note that NLS strategy can also be applied to SparsePEFT.

### 3.3 Leverage the Fine-tuned Super-Adapter

#### 3.3.1 Extract a Sub-Adapter

After completing the fine-tuning stage and obtaining an NLS super-adapter, the next step is extract a desired sub-adapter. The following command demonstrates how to extract the heuristic sub-adapter.
**Additionally, more powerful sub-adapters can be obtained through the advanced search algorithms.** (More details can be found in [here](#333-search-the-optimal-sub-adapter-configuration))

```bash
curl http://<your ip>:8015/v1/sqft/extract_sub_adapter \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "fine_tuning_job_id": <fine tuning job id>,
    "adapter_version": "heuristic"
  }'
```

`adapter_version` can be heuristic, minimal, or a custom name.
When `adapter_version` is set to a custom name, we need to provide a specific configuration in `custom_config`.
The extracted adapter will be saved in `<path to the output directory for this job> / <adapter_version>`.

<details>
<summary>An example of a custom configuration</summary>

```bash
curl http://<your ip>:8015/v1/sqft/extract_sub_adapter \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "fine_tuning_job_id": <fine tuning job id>,
    "adapter_version": "optimal",
    "custom_config": [8, 16, 8, 12, 16, 12, 12, 12, 12, 12, 8, 12, 12, 12, 12, 12]
  }'
```

In the fine-tuning job with the Neural Low-rank adapter Search algorithm,  the `elastic_adapter_config.json` file (which includes the elastic adapter information) will be saved in the job's output directory.
The `custom_config` must correspond with the `target` (adapter modules) or `search_space`
(search space for the rank of adapter modules) in `elastic_adapter_config.json`. 
In the NLS example [here](#321-neural-lora-search-nls), the custom config in the above command `[8, 16, 8, 12, 16, 12, 12, 12, 12, 12, 8, 12, 12, 12, 12, 12]` represents the LoRA rank size of the adapters for `q_proj`, `k_proj`, and `v_proj` in each layer.
It will save the sub-adapter to `<path to the output directory for this job> / optimal`.

</details>

#### 3.3.2 Merge Adapter to Base Model

The following command demonstrates how to merge a sub-adapter (using the Heuristic sub-adapter as an example) into the base pre-trained model to obtain the final fine-tuned model:

```bash
curl http://<your ip>:8015/v1/sqft/merge_adapter \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "fine_tuning_job_id": <fine tuning job id>,
    "adapter_version": "heuristic"
  }'
```

The merged model will be saved in `<path to the output directory for this job> / <adapter_version> / merged_model`.


#### 3.3.3 Search the Optimal Sub-Adapter Configuration

To further explore for high-performing sub-adapter configurations within the super-adapter, we can utilize more advanced search algorithms to search the super-adapter.
Due to the flexibility and wide range of choices in the search settings, the service does not support the search process (but it supports providing a specific sub-adapter configuration to extract the sub-adapter; refer to [here](#331-extract-a-sub-adapter)).
The search needs to be conducted service-externally according to user preferences.

In our example, we provide a simple script ([search.py](./search.py)) for the search (hill-climbing algo) with Arc-E validation set to obtain some optimal sub-adapters.
The command is as follows:

```bash
python search.py \
  --base_model_path meta-llama/Llama-3.2-1B \
  --super_adapter_model_path <path to super adapter> \
  --nls_target_modules q_proj k_proj v_proj \
  --search_space 16 12 8
```


## Toy Experiment Results (NLS)

- w/o tuning

```bash
lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.2-1B --tasks arc_easy --batch_size 32

| Tasks  |Version|Filter|n-shot| Metric |Value |   |Stderr|
|--------|------:|------|-----:|--------|-----:|---|-----:|
|arc_easy|      1|none  |     0|acc     |0.6528|±  |0.0098|
|        |       |none  |     0|acc_norm|0.6065|±  |0.0100|
```

- with vanilla LoRA tuning

```bash
lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.2-1B,peft=<path to adapter> --tasks arc_easy --batch_size 32

| Tasks  |Version|Filter|n-shot| Metric |Value |   |Stderr|
|--------|------:|------|-----:|--------|-----:|---|-----:|
|arc_easy|      1|none  |     0|acc     |0.6894|±  |0.0095|
|        |       |none  |     0|acc_norm|0.6852|±  |0.0095|
```

- with NLS tuning (heuristic, no search)

```bash
lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.2-1B,peft=<path to heuristic adapter> --tasks arc_easy --batch_size 32

| Tasks  |Version|Filter|n-shot| Metric |Value |   |Stderr|
|--------|------:|------|-----:|--------|-----:|---|-----:|
|arc_easy|      1|none  |     0|acc     |0.6911|±  |0.0095|
|        |       |none  |     0|acc_norm|0.6881|±  |0.0095|
```

- with NLS tuning (search)

```bash
lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.2-1B,peft=<path to search optimal adapter> --tasks arc_easy --batch_size 32

| Tasks  |Version|Filter|n-shot| Metric |Value |   |Stderr|
|--------|------:|------|-----:|--------|-----:|---|-----:|
|arc_easy|      1|none  |     0|acc     |0.6911|±  |0.0095|
|        |       |none  |     0|acc_norm|0.6902|±  |0.0095|
```
