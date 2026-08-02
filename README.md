# vllm_sh_To_yaml

##Features:
- Convert vLLM launch shell script to vLLM-Ascend yaml

## Usage:

pip install -r requirements.txt

## 单机混部 包含精度和性能场景
python3 main.py -i vllm.sh -o vllm_single.yaml --template single --num_nodes 1 --npu_per_node 16 --benchmark acc,perf

## 多机混部 包含精度和性能场景
python3 main.py -i vllm_qwen3.sh -o vllm_qwen3.yaml --template multi_mix --num_nodes 2 --npu_per_node 16 --benchmark acc,perf

## 分布式PD分离
