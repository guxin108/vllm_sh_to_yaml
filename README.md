# vllm_sh_To_yaml

##Features:
- Convert vLLM launch shell script to vLLM-Ascend yaml

## 使用方法:
cd vllm_sh_to_yaml

pip install -r requirements.txt

### A3单机混部 包含精度和性能场景
python3 main.py -i vllm.sh -o vllm_single.yaml --template single --num_nodes 1 --npu_per_node 16 --benchmark acc,perf

### A3多机混部 包含精度和性能场景
python3 main.py -i vllm_qwen3.sh -o vllm_qwen3.yaml --template multi_mix --num_nodes 2 --npu_per_node 16 --benchmark acc,perf

### 分布式PD分离
