from jinja2 import Environment, FileSystemLoader

def generate_yaml(data, output):
    env = Environment(loader=FileSystemLoader("templates"), trim_blocks=True, lstrip_blocks=True)
    template = env.get_template(f"{data.get('template','single')}.yaml")
    result = template.render(**data)
    lines = [x.rstrip() for x in result.splitlines() if x.strip()]
    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
