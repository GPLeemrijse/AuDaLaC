from argparse import ArgumentParser
import re

class Node:
	def __init__(self, idx, prio, owner):
		self.idx = idx;
		self.prio = prio;
		self.owner = owner;


nodes = [];
edges = [];
regex = re.compile(r"([0-9]+) ([0-9]+) ([0-9]+) ([0-9]+(,[0-9]+)*);");
prios = [0, 0, 0, 0, 0];


def edge_inst(u, v, idx):
	return f"{u} {v} {idx + 2} 1\n";

def main():
	parser = ArgumentParser(prog='pgsolver2adl')
	parser.add_argument('pg_file', help="The pgsolver input file.");
	parser.add_argument('output', help="The output file.");
	args = parser.parse_args()

	with open(args.pg_file, "r") as in_file:
		lines = in_file.readlines();
		for l in lines[1:]: # skip header
			m = regex.match(l);
			if m is None:
				exit(f"Error: no match ({l})");
			u = int(m.group(1)) + 1; # We use 1-indexed id's
			prio = int(m.group(2));
			owner = 1 if (m.group(3) == "1") else 0;
			successors = [(int(v)+1) for v in m.group(4).split(",")]; # We use 1-indexed id's
			assert(prio <= 4);
			prios[prio] += 1;
			nodes.append(Node(u, prio, owner));
			for v in successors:
				edges.append((u, v));

	# Print .init file
	with open(args.output, "w") as out_file:
		nrof_nodes = len(nodes);
		nrof_edges = len(edges);
		nrof_measures = nrof_nodes + nrof_edges + 1;
		out_file.writelines([
			"ADL structures 3\n",
			"Edge Node Node Measure Measure\n",
			"Measure Bool Nat Nat\n",
			"Node Nat Bool Measure Measure Measure\n",
			f"Edge instances {nrof_edges}\n"
		]);

		out_file.writelines([edge_inst(u, v, idx) for (idx, (u, v)) in enumerate(edges)]);

		out_file.writelines([
			f"Measure instances {nrof_measures}\n",
		]);
		# max
		out_file.write("0 %s\n" % " ".join(str(p) for (d, p) in enumerate(prios) if d % 2 == 1));

		out_file.writelines(["0 0 0\n"] * (nrof_measures - 1));

		out_file.writelines([
			f"Node instances {nrof_nodes}\n",
		]);

		out_file.writelines([f"{n.prio} {n.owner} {idx+nrof_edges+2} 0 1\n" for (idx, n) in enumerate(nodes)]);

if __name__ == '__main__':
	main()

