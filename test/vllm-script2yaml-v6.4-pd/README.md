vllm-script2yaml-v6.4-pd

支持：
- single
- multi_mix
- multi_pd

PD规则：
- producer/consumer 从 kv_role 自动识别
- engine_id 自动递增
- kv_port producer/consumer 组内保持一致
- dp/tp config:
  producer -> kv_connector_extra_config.prefill
  consumer -> kv_connector_extra_config.decode
