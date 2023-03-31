import sys
import random
import os
import re
import unittest

# Read parameters
n = int(sys.argv[1]); # node
m_exp = int(sys.argv[2]); # expected number of edges


assert m_exp <= (n*n);
assert m_exp > 0;
assert n > 0;

p = m_exp/(n*n); # probability of edge between any given pair of nodes


# Generate random graph
graph = [];
for u in range(n):
    for v in range(n):
        if random.random() < p:
            graph.append((u, v));

m = len(graph);


# Print as simple input file
with open("/tmp/random.in", "w") as graph_f:
    graph_f.write(f"{n} {m}\n");

    for (u, v) in graph:
        graph_f.write(f"{u} {v}\n");

# Print as ADL init file
with open("/tmp/random.init", "w") as graph_f:
    graph_f.write("ADL structures 3\nEdge Node Node\nNode NodeSet Bool Bool\nNodeSet Node Bool NodeSet NodeSet NodeSet\n");
    graph_f.write(f"Edge instances {m}\n");

    for (u, v) in graph:
        graph_f.write(f"{u+1} {v+1}\n");

    graph_f.write(f"Node instances {n}\n");
    nodes = "1 0 0\n" * n;
    graph_f.write(nodes);
    graph_f.write("NodeSet instances 1\n0 0 0 0 0\n");

# Run on cpp version
stream = os.popen('cat /tmp/random.in | ../comparisons/kosarajus');
output = stream.read().split("\n");
status = stream.close();
if status:
	print("CPP error: ", end="");
	print(os.waitstatus_to_exitcode(status));
lines_cpp = [];


# Make standard output format
for l in output[:-2]: # Skip trailing nl and nrof components
	nodes = [int(node) for node in l.strip().split(" ")];
	lines_cpp.append(" ".join([str(i) for i in sorted(nodes)]));
lines_cpp.sort();

# Run on ADL version
stream = os.popen('../SCC.out /tmp/random.init');
output = stream.read().split("\n");
status = stream.close();
if status:
	print("ADL error: ", end="");
	print(os.waitstatus_to_exitcode(status));


# Make standard output format
regex = re.compile(r"^Node\(([0-9]+)\): set=([0-9]+)");
lines_adl = [];
sccs = {};
for l in output:
	match = regex.match(l);
	if match:
		node_num = match.group(1);
		scc = match.group(2);
		if scc in sccs:
			sccs[scc].append(int(node_num)-1);
		else:
			sccs[scc] = [int(node_num)-1];

for scc in sccs.keys():
	lines_adl.append(" ".join(str(i) for i in sorted(sccs[scc])));

lines_adl.sort();


if lines_adl != lines_cpp:
	for (l1, l2) in zip(lines_adl, lines_cpp):
		if l1 != l2:
			print("Differ:");
			print(l1);
			print(l2);
			break;
	with open("ADL.output", "w") as adl_out:
		adl_out.write("\n".join(lines_adl));
	with open("CPP.output", "w") as cpp_out:
		cpp_out.write("\n".join(lines_cpp));


else:
	print("Equal!");
