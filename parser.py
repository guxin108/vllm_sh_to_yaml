import re
import shlex

IGNORE_ENV = {"LD_PRELOAD"}

IGNORE_ARGS = {
    "--host",
    "--served-model-name",
}


def parse_envs(content):
    envs = {
        "SERVER_PORT": "DEFAULT_PORT"
    }

    for k, v in re.findall(r"export\s+(\w+)=(.*)", content):
        if k not in IGNORE_ENV:
            envs[k] = v.strip().strip('"').strip("'")

    return envs


def parse_model(content):
    m = re.search(r"vllm\s+serve\s+([^\s\\]+)", content)
    if not m:
        raise RuntimeError("model not found")
    return m.group(1)


def split_command(content):
    start = content.find("vllm serve")
    if start < 0:
        return []

    cmd = content[start:].replace("\\\n", " ")

    lexer = shlex.shlex(cmd, posix=False)
    lexer.whitespace_split = True

    return list(lexer)


def parse_server_cmd(content):
    tokens = split_command(content)
    args = tokens[3:]

    result = []
    i = 0

    while i < len(args):
        t = args[i]

        if t in IGNORE_ARGS:
            i += 2
            continue

        if t == "--port":
            result.extend(["--port", "$SERVER_PORT"])
            i += 2
            continue

        result.append(t)

        if t.startswith("--") and i + 1 < len(args) and not args[i + 1].startswith("--"):
            result.append(args[i + 1])
            i += 2
        else:
            i += 1

    return result


def parse_raw_server_cmd(content):
    """Keep original vllm serve command for multi-node templates."""
    start = content.find("vllm serve")
    if start < 0:
        raise RuntimeError("vllm serve command not found")

    cmd = content[start:].strip()
    cmd = cmd.replace("\\\n", "\n")

    tokens = split_command(content)
    args = tokens[3:]
    result = []
    i = 0

    while i < len(args):
        t = args[i]
        if t in IGNORE_ARGS:
            i += 2
            continue
        result.append(t)
    return cmd,result


def parse_name(model):
    return model.split("/")[-1]
