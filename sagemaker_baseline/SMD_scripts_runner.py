import os
import json
import sys
import time
import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlow
from sagemaker.debugger import Rule, DebuggerHookConfig, CollectionConfig, rule_configs

if __name__ == "__main__":
    client = boto3.client('sagemaker')
    session = boto3.session.Session()
    # This separation of rules into three subsets was required to run a max of 4 Processing jobs on the selected machine ml.p2.xlarge
    first_built_in_rules = [
        Rule.sagemaker(rule_configs.weight_update_ratio()),
        Rule.sagemaker(rule_configs.vanishing_gradient()),
        Rule.sagemaker(rule_configs.loss_not_decreasing()),
        Rule.sagemaker(base_config=rule_configs.dead_relu())
    ]
    second_built_in_rules = [
        Rule.sagemaker(base_config=rule_configs.poor_weight_initialization()),
        Rule.sagemaker(base_config=rule_configs.exploding_tensor(), 
                       rule_parameters={"collection_names": "weights, biases"},
                       collections_to_save=[ 
                                            CollectionConfig(
                                                name="weights", 
                                                parameters={
                                                    "save_interval": "500"
                                                }),
                                            CollectionConfig(
                                                name="biases", 
                                                parameters={
                                                    "save_interval": "500"
                                                })]),
        Rule.sagemaker(base_config=rule_configs.tensor_variance(),
                       rule_parameters={
                                "collection_names": "weights",
                                "max_threshold": "10",
                                "min_threshold": "0.00001",
                        }),
        Rule.sagemaker(base_config=rule_configs.all_zero(), 
                       rule_parameters={"collection_names": "weights"},
                       collections_to_save=[ 
                                            CollectionConfig(
                                                name="weights", 
                                                parameters={
                                                    "save_interval": "500"
                                                })
                                           ]),
    ]
    third_built_in_rules = [
      Rule.sagemaker(rule_configs.overfit()),
      Rule.sagemaker(rule_configs.overtraining()),
      Rule.sagemaker(
        base_config=rule_configs.unchanged_tensor(),
        rule_parameters={
                "collection_names": "losses",
                "tensor_regex": "",
                "num_steps": "3",
                "rtol": "1e-05",
                "atol": "1e-08",
                "equal_nan": "False"
        },
        collections_to_save=[ 
            CollectionConfig(
                name="losses", 
                parameters={
                    "save_interval": "500"
                } 
            )
        ]) 
      Rule.sagemaker(
            base_config=rule_configs.check_input_images(),
            rule_parameters={
                    "threshold_mean": "0.2",
                    "threshold_samples": "500",
                    "regex": "inputs",
                    "channel": "1"
            },
            collections_to_save=[ 
                CollectionConfig(
                    name="custom_collection_inputs", 
                    parameters={
                        "include_regex": "inputs",
                        "save_interval": "500"
                    } 
                )
            ]
        )
    ]
    built_in_rules = [first_built_in_rules, second_built_in_rules, third_built_in_rules]
    results = {}
    directory = r'./scripts_dir'
    for filename in os.listdir(directory):
        if filename.endswith(".py"):
            model_name = filename.replace('.py','').replace('_','-')
            results[model_name] = []
            for built_in_rule in built_in_rules:
                estimator = TensorFlow(
                    base_job_name=model_name,
                    role=sagemaker.get_execution_role(),
                    instance_count=1,
                    instance_type="ml.p2.xlarge",
                    framework_version='1.15',
                    py_version="py36",
                    max_run=3600,
                    source_dir=directory,
                    entry_point=filename,
                    rules=built_in_rule,
                    disable_profiler=True,
                )
                estimator.fit()
