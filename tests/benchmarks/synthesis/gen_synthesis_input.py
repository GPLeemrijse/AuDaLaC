import sys
import os
import networkx as nx
import random
import itertools
import numpy as np
from argparse import ArgumentParser
#G = nx.Graph()

def is_positive_value(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not > 0")
    return ivalue

def main():
	parser = ArgumentParser(prog='gen_synthesis_input')
	parser.add_argument('N', type=is_positive_value, nargs='+', help="The lengths of the generated lists.");
	parser.add_argument('output_dir', help="The output directory.");
	args = parser.parse_args()

	for n in args.N:
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

		p_initial = 0.05;
		p_marked = 0.2;

		permuted_edges = np.random.permutation(graph.edges);
		controllable_frac = 0.6;
		nrof_controllable = int(m * controllable_frac);

		# Write synthesis.adl init file to disk
		adl_init_file = os.path.join(args.output_dir, f"synthesis_{n}_{m}.init");
		with open(adl_init_file, "w") as graph_f:
			graph_f.write("ADL structures 3\n"
				"State Bool Bool Bool Bool Bool Bool\n"
				"ControllableEvent State State\n"
				"UncontrollableEvent State State\n");
			
			# Print nodes
			graph_f.write(f"State instances {n} {n}\n");
			for i in range(n):
				p = random.random();
				is_init = p < p_initial;
				is_marked = p < p_marked;
				graph_f.write(f"{int(is_marked)} {int(is_init)} 1 0 0 0\n");

			# Print ControllableEvent
			graph_f.write(f"ControllableEvent instances {nrof_controllable} {nrof_controllable}\n");
			for (u, v) in permuted_edges[:nrof_controllable]:
				graph_f.write(f"{u+1} {v+1}\n");

			# Print UncontrollableEvent
			graph_f.write(f"UncontrollableEvent instances {n - nrof_controllable} {n - nrof_controllable}\n");
			for (u, v) in permuted_edges[nrof_controllable:]:
				graph_f.write(f"{u+1} {v+1}\n");


if __name__ == '__main__':
	main()