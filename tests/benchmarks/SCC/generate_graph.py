import sys
import os
import networkx as nx
G = nx.Graph()


n = int(sys.argv[1]); # node
out_dir = sys.argv[2]; # Output directory

p = 1.3/n;

m_exp = p*n*n;

cont = input(f"Expected nrof edges is {m_exp:,} and n*p = {n*p}, do you wish to continue? [y/n]");
if cont != "y":
	exit("Stopped.");

use_fast = n**2 > (n + p*n*n); # O(n^2) vs O(n + m)

if use_fast:
	print("Using fast!");
	graph = nx.fast_gnp_random_graph(n, p, directed=True);
else:
	print("Using default!");
	graph = nx.gnp_random_graph(n, p, directed=True);

m = graph.number_of_edges();

print(f"The graph contains {n} nodes and {m} edges.");

# Write input file for sequential to disk
sequential_in_file = os.path.join(out_dir, f"random_kos_{n}_{m}.in");
with open(sequential_in_file, "w") as graph_f:
	graph_f.write(f"{n} {m}\n");
	for (u, v) in graph.edges:
		graph_f.write(f"{u} {v}\n");


# Write SCC.adl ADL init file to disk
adl_init_file = os.path.join(out_dir, f"random_scc_{n}_{m}.init");
with open(adl_init_file, "w") as graph_f:
	graph_f.write("ADL structures 3\nEdge Node Node\nNode NodeSet Bool Bool\nNodeSet Node Node Node Node Bool NodeSet NodeSet NodeSet\n");
	graph_f.write(f"Edge instances {m} {m}\n");

	for (u, v) in graph.edges:
		graph_f.write(f"{u+1} {v+1}\n");

	graph_f.write(f"Node instances {n} {n}\n");
	nodes = "1 0 0\n" * n;
	graph_f.write(nodes);
	graph_f.write("NodeSet instances 1 1\n0 0 0 0 0 0 0 0\n");

# # Write SCC_MP.adl ADL init file to disk
# adl_init_file = os.path.join(out_dir, f"random_mp_{n}_{m}.init");
# with open(adl_init_file, "w") as graph_f:
# 	graph_f.write("ADL structures 2\nEdge Node Node\nNode Bool Node Node Bool Bool\n");
# 	graph_f.write(f"Edge instances {m} {m}\n");

# 	for (u, v) in graph.edges:
# 		graph_f.write(f"{u+1} {v+1}\n");

# 	graph_f.write(f"Node instances 0 {n}\n");


# Write SCC_col.adl ADL init file to disk
adl_init_file = os.path.join(out_dir, f"random_col_{n}_{m}.init");
with open(adl_init_file, "w") as graph_f:
	graph_f.write("ADL structures 2\nEdge Node Node Bool\nNode Node Node\n");
	graph_f.write(f"Edge instances {m} {m}\n");

	for (u, v) in graph.edges:
		graph_f.write(f"{u+1} {v+1} 0\n");

	graph_f.write(f"Node instances 0 {n}\n");