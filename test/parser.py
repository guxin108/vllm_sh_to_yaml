import re
import shlex
import json

IGNORE_ENV = {
    "LD_PRELOAD",
    "GLOO_SOCKET_IFNAME",
    "TP_SOCKET_IFNAME",
    "HCCL_IF_IP",
    "HCCL_SOCKET_IFNAME",
    "LD_LIBRARY_PATH",
}

IGNORE_ARGS = {
    "--served-model-name",
}

RUNTIME_ARGS = {
    "--data-parallel-size": "${DP_SIZE}",
    "--data-parallel-rank": "${DP_RANK}",
    "--data-parallel-address": "${DP_ADDRESS}",
    "--data-parallel-rpc-port": "${DP_RPC_PORT}",
    "--tensor-parallel-size": "${TP_SIZE}",
}

def parse_envs(content):
    env = {"SERVER_PORT":"DEFAULT_PORT"}

    SPECIAL_ENV = {
        "ASCEND_RT_VISIBLE_DEVICES": "${VISIBLE_DEVICES}",
        "SERVER_PORT": "${PORT}",
    }

    for k,v in re.findall(r"export\s+(\w+)=(.*)", content):
        if k in IGNORE_ENV:
            continue

        if k in SPECIAL_ENV:
            env[k] = SPECIAL_ENV[k]
            continue

        env[k] = v.strip().strip('"').strip("'")

    return env

def parse_model(content):
    m=re.search(r"vllm\s+serve\s+([^\s\\]+)",content)
    if not m:
        raise RuntimeError("model not found")
    return m.group(1)

def parse_name(model):
    return model.split("/")[-1]

def split_command(content):
    start=content.find("vllm serve")
    if start<0:
        return []
    cmd=content[start:].replace("\\\n"," ")
    return shlex.split(cmd)

def parse_server_cmd(content):
    tokens=split_command(content)
    args=tokens[3:]
    out=[]
    i=0

    while i<len(args):
        t=args[i]

        if t in IGNORE_ARGS:
            i+=2
            continue

        if t == "--port":
            out.extend([t, "$SERVER_PORT"])
            i += 2
            continue

        if t in RUNTIME_ARGS:
            out.extend([t,RUNTIME_ARGS[t]])
            i+=2
            continue

        out.append(t)

        if t.startswith("--") and i+1<len(args) and not args[i+1].startswith("--"):
            out.append(args[i+1])
            i+=2
        else:
            i+=1

    return out

def parse_kv_transfer(content):
    m=re.search(
        r"--kv-transfer-config\s+\\?\s*'(\{.*?\})'",
        content,
        re.S
    )
    if not m:
        return {}
    raw=m.group(1).replace("\n"," ")
    return json.loads(raw)

def parse_role(content):
    kv=parse_kv_transfer(content)
    return kv.get("kv_role")

def parse_pd_config(prefill, decode, p_nodes, d_nodes, npu):
    pk=parse_kv_transfer(prefill)
    dk=parse_kv_transfer(decode)

    p=pk.get("kv_connector_extra_config",{}).get("prefill",{})
    d=dk.get("kv_connector_extra_config",{}).get("decode",{})

    result=[]
    # prefill nodes
    for i in range(p_nodes):
        tp=p.get("tp_size")
        local=npu//tp
        result.append({
            "node_index": i,
            "dp_size": p.get("dp_size"),
            "dp_size_local": local,
            "dp_rank_start": i*local,
            "tp_size": tp,
            "dp_address": f"${{NODE_0_IP}}"
        })
    # decode nodes
    for j in range(d_nodes):
        idx=p_nodes+j
        tp=d.get("tp_size")
        local=npu//tp
        result.append({
            "node_index": idx,
            "dp_size": d.get("dp_size"),
            "dp_size_local": local,
            "dp_rank_start": j*local,
            "tp_size": tp,
            "dp_address": f"${{NODE_{p_nodes}_IP}}"
        })
    return result


def replace_runtime_args(cmd):
    out=[]
    i=0
    while i < len(cmd):
        if cmd[i] == "--port" and i+1 < len(cmd):
            out += [cmd[i], "$SERVER_PORT"]; i+=2; continue
        if cmd[i] in RUNTIME_ARGS and i+1 < len(cmd):
            out += [cmd[i], RUNTIME_ARGS[cmd[i]]]; i+=2; continue
        out.append(cmd[i]); i+=1
    return out


def patch_kv_config(cmd, engine_id):
    out=[]
    for x in cmd:
        if x.startswith("{") and "kv_connector" in x:
            try:
                data=json.loads(x)
                data["engine_id"]=str(engine_id)
                x=json.dumps(data,separators=(", ", ": "))
            except Exception:
                pass
        out.append(x)
    return out


def format_cmd_value(value):
    """Format yaml list value.
    Runtime distributed args keep ${} unquoted, all other values quoted.
    JSON objects use single quotes.
    """
    runtime = {
        "$SERVER_PORT",
        "${DP_SIZE}",
        "${DP_RANK}",
        "${DP_ADDRESS}",
        "${DP_RPC_PORT}",
        "${TP_SIZE}",
    }
    if value in runtime:
        return value
    if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
        return "'" + value.replace("'", "\\'") + "'"
    return '"' + str(value).replace('"', '\\"') + '"'
