import math

problem_sizes = [2**n for n in range(5, 17)];

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

    init < Fix(reachability)
    """;

def generate_body(n, m):
    indent = "            ";
    result = "";

    for y in range(n):
        for x in range(m):
            result += f"{indent}Node n_{x}_{y} := Node({'true' if y==0 else 'false'});\n";
            if y != 0:
                result += f"{indent}Edge e_{x}_{y} := Edge(n_{x}_{y-1}, n_{x}_{y});\n";
    return result;


def generate_n_by_m_program(n, m):
    return format_program(generate_body(n, m));

# for i in problem_sizes:
#     for j in range(0, round(math.log2(i))):
#         n = round(i / (2**j));
#         m = 2**j;
for n in [2**n for n in range(0, 8)]:
    for m in [2**n for n in range(1, 8)]:
        program = generate_n_by_m_program(n, m);
        name = f"reachability_{n*m}_{n}_{m}.adl";
        with open(name, 'w') as f:
            f.write(program);