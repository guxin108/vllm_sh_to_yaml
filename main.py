import argparse
import yaml

from parser import parse_envs, parse_model, parse_server_cmd, parse_raw_server_cmd, parse_name
from yaml_generator import generate_yaml


def load_benchmarks(path, selected):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    result = {}
    for name in [x.strip() for x in selected.split(",")]:
        if name not in cfg:
            raise ValueError(f"Unknown benchmark: {name}")
        result[name] = cfg[name]

    return result


def main():
    ap = argparse.ArgumentParser(description="Convert vLLM shell script to vLLM-Ascend YAML")
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", default="generated.yaml")
    ap.add_argument("--benchmark", default="perf")
    ap.add_argument("--benchmark-config", default="benchmark.yaml")
    ap.add_argument("--template", choices=["single", "multi_mix"], default="single")
    ap.add_argument("--num_nodes", type=int, default=1)
    ap.add_argument("--npu_per_node", type=int, default=8)

    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        content = f.read()

    model = parse_model(content)

    # multi_mix keeps the original command. single keeps v6 parsing behavior.
    if args.template == "multi_mix":
        server_cmd = parse_raw_server_cmd(content)
    else:
        server_cmd = parse_server_cmd(content)

    data = {
        "name": parse_name(model),
        "model": model,
        "envs": parse_envs(content),
        "server_cmd": server_cmd,
        "benchmarks": load_benchmarks(args.benchmark_config, args.benchmark),
        "template": args.template,
        "num_nodes": args.num_nodes,
        "npu_per_node": args.npu_per_node,
    }

    generate_yaml(data, args.output)
    print("Generated:", args.output)


if __name__ == "__main__":
    main()
