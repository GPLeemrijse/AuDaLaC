import math

problem_sizes = [2**n for n in range(7, 15)];

def format_program(body):
    return f"""
    struct Node (reachable : Bool) {{
        init {{
            {body}
        }}
    }}

    struct Edge (n1 : Node, n2 : Node) {{
        reachability {{
            if n1.reachable then {{
                n2.reachable := true;
            }}
        }}
    }}

    init < Fix(reachability) < Node.print
    """;

def generate_body(n, m):
    indent = "            ";
    result = f"Node s := Node(true);\n{indent}Node t := Node(false);\n\n";

    for y in range(n):
        for x in range(m):
            result += f"{indent}Node n_{x}_{y} := Node(false);\n";
            prev = "s" if y == 0 else f"n_{x}_{y-1}";
            result += f"{indent}Edge e_{x}_{y} := Edge({prev}, n_{x}_{y});\n";
            if y == n - 1:
                result += f"{indent}Edge e_{x}_{y}_target := Edge(n_{x}_{y}, t);\n";
    return result;


def generate_n_by_m_program(n, m):
    return format_program(generate_body(n, m));

for i in problem_sizes:
    for j in range(0, round(math.log2(i))+1):
        n = round(i / (2**j));
        m = 2**j;
        program = generate_n_by_m_program(n, m);
        name = f"reachability_{n}_{m}.adl";
        with open(name, 'w') as f:
            f.write(program);