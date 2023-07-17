import random
import os
from argparse import ArgumentParser

def is_positive_value(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not > 0")
    return ivalue

def main():
	parser = ArgumentParser(prog='sorting_list_generator')
	parser.add_argument('N', type=is_positive_value, nargs='+', help="The lengths of the generated lists.");
	parser.add_argument('output_dir', help="The output directory.");
	args = parser.parse_args()

	for n in args.N:
		nums = random.sample(range(-1000000, 1000000), n);

		output_file_name = os.path.join(args.output_dir, f"sorting_{n}.init");

		with open(output_file_name, "w") as out_file:
			out_file.writelines([
				"ADL structures 2\n",
				"ListElem Int ListElem ListElem ListElem Bool\n",
				"Printer ListElem\n",
				f"ListElem instances {n} {n}\n"
			]);

			out_file.writelines([f"{val} {idx+2 if idx != n - 1 else 0} 0 1 0\n" for (idx, val) in enumerate(nums)]);

			out_file.writelines(["Printer instances 0 0\n"]);

if __name__ == '__main__':
	main()
