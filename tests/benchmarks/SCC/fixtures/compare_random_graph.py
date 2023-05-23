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


random_nrs = [];

for i in range(200):
	random_nrs.append(random.random());

i = 0;
i_mod_200 = 0;
i_mod_10000 = 0;
# Generate random graph
graph = [];
for u in range(n):
    for v in range(n):
        if random_nrs[i_mod_200] < p:
            graph.append((u, v));
        i += 1;
        i_mod_200 += 1;
        i_mod_10000 += 1;
        if i_mod_200 == 200:
        	i_mod_200 = 0;
        if i_mod_10000 == 10000:
        	print(f"\rGenerated {int(i/100000)}/{int(n*n/100000)} * 10^5 possible edges", end="");
        	i_mod_10000 = 0;

m = len(graph);


# Print as simple input file
with open("/tmp/random.in", "w") as graph_f:
    graph_f.write(f"{n} {m}\n");

    for (u, v) in graph:
        graph_f.write(f"{u} {v}\n");

# Print as ADL init file
with open("/tmp/random.init", "w") as graph_f:
    graph_f.write("ADL structures 3\nEdge Node Node\nNode NodeSet Bool Bool\nNodeSet Node Node Node Node Bool NodeSet NodeSet NodeSet\n");
    graph_f.write(f"Edge instances {m}\n");

    for (u, v) in graph:
        graph_f.write(f"{u+1} {v+1}\n");

    graph_f.write(f"Node instances {n}\n");
    nodes = "1 0 0\n" * n;
    graph_f.write(nodes);
    graph_f.write("NodeSet instances 1\n0 0 0 0 0 0 0 0\n");


# Run on ADL version
stream = os.popen('../SCC.out /tmp/random.init');
#stream = os.popen('../SCC.out SCC_2.init');
status = stream.close();
if status:
	print("ADL error: ", end="");
	print(os.waitstatus_to_exitcode(status));

# Run on cpp version
stream = os.popen('cat /tmp/random.in | ../comparisons/kosarajus');
#stream = os.popen('cat SCC_3.in | ../comparisons/kosarajus');
output = stream.read().split("\n");
status = stream.close();
if status:
	print("CPP error: ", end="");
	print(os.waitstatus_to_exitcode(status));
lines_cpp = [];

print("Done.");