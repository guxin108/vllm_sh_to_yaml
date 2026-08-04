from jinja2 import Environment, FileSystemLoader

def generate_yaml(data, output):
    env=Environment(
        loader=FileSystemLoader("templates"),
        trim_blocks=True,
        lstrip_blocks=True
    )

    tpl=env.get_template("multi_pd.yaml")
    env.globals["format_cmd_value"] = __import__("parser").format_cmd_value
    result=tpl.render(**data)

    lines=[]
    for line in result.splitlines():
        if line.strip():
            lines.append(line.rstrip())

    with open(output,"w",encoding="utf-8") as f:
        f.write("\n".join(lines)+"\n")
