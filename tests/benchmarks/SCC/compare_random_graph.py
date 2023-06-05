import sys
import random
import os
import re
import unittest

def generate_graph(init_file, in_file):
	# Read parameters
	n = int(sys.argv[1]); # node
	m_exp = int(sys.argv[2]); # expected number of edges


	assert m_exp <= (n*n);
	assert m_exp > 0;
	assert n > 0;

	p = m_exp/(n*n); # probability of edge between any given pair of nodes

	i = 0;
	i_mod_10000 = 0;
	# Generate random graph
	graph = [];
	for u in range(n):
	    for v in range(n):
	        if random.random() < p:
	            graph.append((u, v));
	        i += 1;
	        i_mod_10000 += 1;
	        if i_mod_10000 == 10000:
	        	percentage = round((i*100)/(n*n));
	        	print(f"\rGenerated {percentage}% possible edges", end="");
	        	i_mod_10000 = 0;

	m = len(graph);

	print(f"\nGenerated a graph with {n} nodes and {m} edges.");


	# Print as simple input file
	with open(in_file, "w") as graph_f:
	    graph_f.write(f"{n} {m}\n");
	    for (u, v) in graph:
	        graph_f.write(f"{u} {v}\n");

	# Print as ADL init file
	with open(init_file, "w") as graph_f:
	    graph_f.write("ADL structures 3\nEdge Node Node\nNode NodeSet Bool Bool\nNodeSet Node Node Node Node Bool NodeSet NodeSet NodeSet\n");
	    graph_f.write(f"Edge instances {m} {m}\n");

	    for (u, v) in graph:
	        graph_f.write(f"{u+1} {v+1}\n");

	    graph_f.write(f"Node instances {n} {n}\n");
	    nodes = "1 0 0\n" * n;
	    graph_f.write(nodes);
	    graph_f.write("NodeSet instances 1 1\n0 0 0 0 0 0 0 0\n");


if sys.argv[1] == "-f":
	used_init_file = sys.argv[2] + ".init";
	used_in_file = sys.argv[2] + ".in";
	gen_graph = False;
else:
	used_init_file = "/tmp/random.init";
	used_in_file = "/tmp/random.in";
	gen_graph = True;

if gen_graph:
	generate_graph(used_init_file, used_in_file);

# Run on ADL version
adl_cmd = f"./SCC.out {used_init_file}";
print(f"ADL launch: {adl_cmd}");
stream = os.popen(adl_cmd);
output = stream.read().split("\n");
status = stream.close();
if status:
	print("ADL error: ", end="");
	print(os.waitstatus_to_exitcode(status));

# Run on cpp version
cpp_cmd = f"cat {used_in_file} | comparisons/kosarajus";
print(f"CPP launch: {cpp_cmd}");
stream = os.popen(cpp_cmd);
output = stream.read().split("\n");
status = stream.close();
if status:
	print("CPP error: ", end="");
	print(os.waitstatus_to_exitcode(status));
lines_cpp = [];

print("Done.");