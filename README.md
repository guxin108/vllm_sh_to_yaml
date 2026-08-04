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

 python3 main.py -i test/vllm_qwen3_single.sh -o multi_mix.yaml --template multi_mix --num_nodes 2 --tp 4 --npu_per_node 8

 --data-parallel-size-local的计算规则是--data-parallel-size-local*--tensor-parallel-size=npu_per_node，然后--data-parallel-start-rank也等于--data-parallel-size-local的值
### 分布式PD分离
python3 main.py -p p.sh -d d.sh -o out.yaml --num_nodes 3 --p_nodes 1 --d_nodes 2 --npu_per_node 16
