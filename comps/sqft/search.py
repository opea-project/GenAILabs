# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import importlib.util

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from lm_eval.utils import load_yaml_config


# Get the directory path where the `lm_eval` module is located
spec = importlib.util.find_spec("lm_eval")
module_path = spec.origin
module_dir = os.path.dirname(module_path)

arc_yaml_file = os.path.join(module_dir, "tasks/arc/arc_easy.yaml")
task_config = load_yaml_config(arc_yaml_file)
# Modify the task configuration to define the validation set task
task_config["task"] = "arc_easy_val"
task_config["dataset_name"] = "ARC-Easy"
task_config["test_split"] = "validation"


def main():
    parser = argparse.ArgumentParser(description="Search optimal sub-adapter configuration.")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the base model.")
    parser.add_argument("--super_adapter_model_path", type=str, required=True, help="Path to the super-adapter model.")
    parser.add_argument("--nls_target_modules", type=str, nargs='+', required=True)
    parser.add_argument("--search_space", type=int, nargs='+', required=True)
    args = parser.parse_args()

    base_model_path = args.base_model_path
    super_adapter_model_path = args.super_adapter_model_path
    nls_target_modules = args.nls_target_modules
    search_space = args.search_space

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="float16",
    )
    model = PeftModel.from_pretrained(
        base_model,
        super_adapter_model_path,
        torch_dtype="float16",
        device_map="auto"
    )

    from sqft.elastic_lora_linear import make_lora_elastic
    shared_r_list = make_lora_elastic(
        model, 
        search_space, 
        nls_target_modules
    )
    model.eval()

    def validate_fn(sub_adapte_config):
        # Evaluate the current sub-adapter configuration
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=[task_config],
            batch_size=32,
            log_samples=False,
            task_manager=task_manager,
            **request_caching_args,
        )
        accuracy = results["results"]["arc_easy_val"]["acc_norm,none"]
        print(f"Sub-adapter Config: {sub_adapte_config}")
        print(f"Sub-adapter Accuracy (val): {accuracy}")
        return accuracy

    lm = HFLM(model, batch_size=32, trust_remote_code=True)
    task_manager = TaskManager("INFO", include_path=None)
    request_caching_args = {'cache_requests': False, 'delete_requests_cache': False, 'rewrite_requests_cache': False}
    
    def activate_sub_adapter(shared_r_list, sub_adapte_config):
        assert len(shared_r_list) == len(sub_adapte_config)
        for i in range(len(sub_adapte_config)):
            shared_r = shared_r_list[i]
            value = sub_adapte_config[i]
            shared_r.set_r(value)

    heuristic_config = [shared_r.r_values[len(shared_r.r_values) // 2] for shared_r in shared_r_list]
    activate_sub_adapter(shared_r_list, heuristic_config)
    heu_eval_acc = validate_fn(heuristic_config)
    print(f"Initial Heuristic Accuracy: {heu_eval_acc}")

    def get_neighbors(config, shared_r_list):
        neighbors = []
        for i in range(len(config)):
            current_index = shared_r_list[i].r_values.index(config[i])
            if current_index > 0:
                neighbor = config.copy()
                neighbor[i] = shared_r_list[i].r_values[current_index - 1]
                neighbors.append(neighbor)
            if current_index < len(shared_r_list[i].r_values) - 1:
                neighbor = config.copy()
                neighbor[i] = shared_r_list[i].r_values[current_index + 1]
                neighbors.append(neighbor)
        return neighbors

    current_config = heuristic_config
    current_acc = heu_eval_acc

    while True:
        neighbors = get_neighbors(current_config, shared_r_list)
        best_neighbor = None
        best_acc = current_acc

        for neighbor in neighbors:
            activate_sub_adapter(shared_r_list, neighbor)
            acc = validate_fn(neighbor)
            if acc > best_acc:
                best_acc = acc
                best_neighbor = neighbor

        if best_neighbor is None:
            break

        current_config = best_neighbor
        current_acc = best_acc
        print(f"New Best Config: {current_config} with Accuracy: {current_acc}")

    print(f"Final Best Config: {current_config} with Accuracy: {current_acc}")


if __name__ == "__main__":
    main()
