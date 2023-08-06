from argparse import ArgumentParser, FileType
import re
import os
import numpy as np

class Node:
	def __init__(self, idx, prio, owner):
		self.idx = idx;
		self.prio = prio;
		self.owner = owner;

regex = re.compile(r"([0-9]+) ([0-9]+) ([0-9]+) ([0-9]+(,[0-9]+)*);");


def print_init(output_file_name, nodes, edges, prios, permute):
	if permute:
		edges = np.random.permutation(edges);

	# Print .init file
	with open(output_file_name, "w") as out_file:
		nrof_nodes = len(nodes);
		nrof_edges = len(edges);
		out_file.writelines([
			"ADL structures 2\n",
			"Edge Node Node Nat Nat Bool Nat Nat\n",
			"Node Nat Bool Bool Bool Nat Nat Edge\n",
			f"Edge instances {nrof_edges} {nrof_edges}\n"
		]);

		# Edges
		out_file.writelines([f"{u} {v} {prios[1]} {prios[3]}\n" for (u, v) in edges]);

		# Nodes
		out_file.writelines([
			f"Node instances {nrof_nodes} {nrof_nodes}\n",
		]);
		out_file.writelines([f"{n.prio} {n.owner} 1\n" for (idx, n) in enumerate(nodes)]);


def read_input_file(in_file):
	file_name = in_file.name.split("/")[-1];
	if not file_name.endswith(".gm"):
		print(f"Skipping file {file_name}");
		return ([], [], []);

	nodes = [];
	edges = [];
	prios = [0, 0, 0, 0, 0];
	
	next(in_file); # Skip header
	for l in in_file:
		m = regex.match(l);
		if m is None:
			print(f"Error {file_name}: no match ({l})");
			return ([], [], []);

		u = int(m.group(1)) + 1; # We use 1-indexed id's
		prio = int(m.group(2));
		owner = 1 if (m.group(3) == "1") else 0;
		successors = [(int(v)+1) for v in m.group(4).split(",")]; # We use 1-indexed id's
		
		if prio > 4:
			print(f"Error {file_name}: priority {prio} is > 4.");
			return ([], [], []);

		prios[prio] += 1;
		nodes.append(Node(u, prio, owner));
		for v in successors:
			edges.append((u, v));

	return (nodes, edges, prios);

def main():
	parser = ArgumentParser(prog='pgsolver2adl')
	parser.add_argument('pg_files', type=FileType('r'), nargs='+', help="The pgsolver input file(s).");
	parser.add_argument('output_dir', help="The output directory.");
	parser.add_argument('-f', action='store_true', dest='force', help="Do not confirm before generating.");
	parser.add_argument('-p', action='store_true', dest='permute', help="Permute edges.");



	args = parser.parse_args()

	for in_file in args.pg_files:
		name = in_file.name.split("/")[-1];
		print(f"Loading parity game {name}...");
		if not args.force:
			cont = input(f"Do you wish to continue? [y/n]");
			if cont != "y":
				continue;

		(nodes, edges, prios) = read_input_file(in_file);

		if (nodes, edges, prios) == ([], [], []):
			continue;
		
		if args.permute:
			output_file_name = os.path.join(args.output_dir, name[:-3] + "_perm.init");
		else:
			output_file_name = os.path.join(args.output_dir, name[:-3] + ".init");
		print_init(output_file_name, nodes, edges, prios, args.permute);

		print(f"Sucessfully generated {output_file_name}");

if __name__ == '__main__':
	main()

