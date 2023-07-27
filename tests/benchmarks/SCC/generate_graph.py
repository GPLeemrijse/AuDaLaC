import sys
import os
import networkx as nx
from argparse import ArgumentParser

def is_positive_value(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not > 0")
    return ivalue

def main():
	parser = ArgumentParser(prog='generate_scc_graph')
	parser.add_argument('N', type=is_positive_value, nargs='+', help="The the sizes of the graphs.");
	parser.add_argument('output_dir', help="The output directory.");
	args = parser.parse_args()

	for n in args.N:
		p = 1.3/n;
		use_fast = n**2 > (n + p*n*n); # O(n^2) vs O(n + m)

		if use_fast:
			graph = nx.fast_gnp_random_graph(n, p, directed=True);
		else:
			graph = nx.gnp_random_graph(n, p, directed=True);

		m = graph.number_of_edges();

		# Write input file for sequential to disk
		sequential_in_file = os.path.join(args.output_dir, f"random_kos_{n}_{m}.in");
		with open(sequential_in_file, "w") as graph_f:
			graph_f.write(f"{n} {m}\n");
			for (u, v) in graph.edges:
				graph_f.write(f"{u} {v}\n");


		# Write SCC.adl ADL init file to disk
		adl_init_file = os.path.join(args.output_dir, f"random_scc_{n}_{m}.init");
		with open(adl_init_file, "w") as graph_f:
			graph_f.write("ADL structures 3\nEdge Node Node\nNode NodeSet Bool Bool\nNodeSet Node Node Node Node Bool NodeSet NodeSet NodeSet\n");
			graph_f.write(f"Edge instances {m} {m}\n");

			for (u, v) in graph.edges:
				graph_f.write(f"{u+1} {v+1}\n");

			graph_f.write(f"Node instances {n} {n}\n");
			for i in range(n):
				graph_f.write("1\n");

			graph_f.write("NodeSet instances 0 1\n");

		print(f"(\"tests/benchmarks/SCC/testcases/random_scc_{n}_{m}.init\", vec![\"-N\".to_string(), \"NodeSet={n}\".to_string()], {n} + {m}),")


		# Write SCC_col.adl ADL init file to disk
		adl_init_file = os.path.join(args.output_dir, f"random_col_{n}_{m}.init");
		with open(adl_init_file, "w") as graph_f:
			graph_f.write("ADL structures 2\nEdge Node Node Bool\nNode Node Node\n");
			graph_f.write(f"Edge instances {m} {m}\n");

			for (u, v) in graph.edges:
				graph_f.write(f"{u+1} {v+1} 0\n");

			graph_f.write(f"Node instances 0 {n}\n");

		print(f"(\"tests/benchmarks/SCC/testcases/random_col_{n}_{m}.init\", Vec::new(), {n} + {m}),")


if __name__ == '__main__':
	main()