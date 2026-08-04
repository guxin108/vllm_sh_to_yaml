import argparse
from parser import *
from yaml_generator import generate_yaml

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("-p","--prefill",required=True)
    ap.add_argument("-d","--decode",required=True)
    ap.add_argument("--num_nodes",type=int,default=4)
    ap.add_argument("--p_nodes",type=int,default=2, help="prefill node count")
    ap.add_argument("--d_nodes",type=int,default=2, help="decode node count")
    ap.add_argument("--npu_per_node",type=int,default=16)
    ap.add_argument("-o","--output",default="generated.yaml")
    args=ap.parse_args()

    p=open(args.prefill).read()
    d=open(args.decode).read()

    model=parse_model(p)

    data={
        "name":parse_name(model),
        "model":model,
        "num_nodes":args.num_nodes,
        "p_nodes":args.p_nodes,
        "d_nodes":args.d_nodes,
        "npu_per_node":args.npu_per_node,
        "config":parse_pd_config(
            p,d,args.p_nodes,args.d_nodes,args.npu_per_node
        ),
         "commands": [
            patch_kv_config(replace_runtime_args(parse_server_cmd(p)), i)
            for i in range(args.p_nodes)
        ] + [
            patch_kv_config(replace_runtime_args(parse_server_cmd(d)), args.p_nodes+i)
            for i in range(args.d_nodes)
        ],
        "prefill_env":parse_envs(p),
        "decode_env":parse_envs(d),
    }

    generate_yaml(data,args.output)

if __name__=="__main__":
    main()
