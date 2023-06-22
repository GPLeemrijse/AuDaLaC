from argparse import ArgumentParser, FileType
import re
import os

class Node:
	def __init__(self, idx, prio, owner):
		self.idx = idx;
		self.prio = prio;
		self.owner = owner;

regex = re.compile(r"([0-9]+) ([0-9]+) ([0-9]+) ([0-9]+(,[0-9]+)*);");


def edge_inst(u, v, idx):
	return f"{u} {v} {idx + 2} 1\n";

def edge_inst_opt(u, v, idx):
	return f"{u} {v} 0 0 0 1\n";

def print_normal_init(output_file_name, nodes, edges, prios):
	# Print .init file
	with open(output_file_name, "w") as out_file:
		nrof_nodes = len(nodes);
		nrof_edges = len(edges);
		nrof_measures = nrof_nodes + nrof_edges + 1;
		out_file.writelines([
			"ADL structures 3\n",
			"Edge Node Node Measure Measure\n",
			"Measure Bool Nat Nat\n",
			"Node Nat Bool Measure Measure Measure\n",
			f"Edge instances {nrof_edges} {nrof_edges}\n"
		]);

		out_file.writelines([edge_inst(u, v, idx) for (idx, (u, v)) in enumerate(edges)]);

		out_file.writelines([
			f"Measure instances 1 {nrof_measures}\n",
		]);

		# max
		out_file.write("0 %s\n" % " ".join(str(p) for (d, p) in enumerate(prios) if d % 2 == 1));

		out_file.writelines([
			f"Node instances {nrof_nodes} {nrof_nodes}\n",
		]);

		out_file.writelines([f"{n.prio} {n.owner} {idx+nrof_edges+2} 0 1\n" for (idx, n) in enumerate(nodes)]);

def print_opt_init(output_file_name, nodes, edges, prios):
	# Print .init file
	with open(output_file_name, "w") as out_file:
		nrof_nodes = len(nodes);
		nrof_edges = len(edges);
		out_file.writelines([
			"ADL structures 3\n",
			"Edge Node Node Bool Nat Nat Measure\n",
			"Measure Bool Nat Nat\n",
			"Node Nat Bool Bool Nat Nat Edge Measure\n",
			f"Edge instances {nrof_edges} {nrof_edges}\n"
		]);

		out_file.writelines([edge_inst_opt(u, v, idx) for (idx, (u, v)) in enumerate(edges)]);

		# Measures: max
		out_file.writelines([
			f"Measure instances 1 1\n",
			"0 %s\n" % " ".join(str(p) for (d, p) in enumerate(prios) if d % 2 == 1)
		]);

		out_file.writelines([
			f"Node instances {nrof_nodes} {nrof_nodes}\n",
		]);

		out_file.writelines([f"{n.prio} {n.owner} 0 0 0 0 1\n" for (idx, n) in enumerate(nodes)]);

def main():
	parser = ArgumentParser(prog='pgsolver2adl')
	parser.add_argument('pg_files', type=FileType('r'), nargs='+', help="The pgsolver input file(s).");
	parser.add_argument('output_dir', help="The output directory.");
	parser.add_argument('-o', action='store_true', dest='opt', help="Use optimised SPM alg.");

	args = parser.parse_args()

	for in_file in args.pg_files:
		nodes = [];
		edges = [];
		prios = [0, 0, 0, 0, 0];
		file_name = in_file.name.split("/")[-1];
		assert(file_name.endswith(".gm"));
		output_file_name = os.path.join(args.output_dir, file_name[:-3] + ".init");
		
		first_line = True;
		error = False;
		for l in in_file:
			if first_line:
				first_line = False;
				continue; # Skip header
			m = regex.match(l);
			if m is None:
				print(f"Error {file_name}: no match ({l})");
				error = True;
				break;
			u = int(m.group(1)) + 1; # We use 1-indexed id's
			prio = int(m.group(2));
			owner = 1 if (m.group(3) == "1") else 0;
			successors = [(int(v)+1) for v in m.group(4).split(",")]; # We use 1-indexed id's
			if prio > 4:
				print(f"Error {file_name}: priority {prio} is > 4.");
				error = True;
				break;
			prios[prio] += 1;
			nodes.append(Node(u, prio, owner));
			for v in successors:
				edges.append((u, v));

		if error:
			print(f"Skipping file {file_name}...");
			continue;
		
		if args.opt:
			print_opt_init(output_file_name, nodes, edges, prios);
		else:
			print_normal_init(output_file_name, nodes, edges, prios);

if __name__ == '__main__':
	main()

