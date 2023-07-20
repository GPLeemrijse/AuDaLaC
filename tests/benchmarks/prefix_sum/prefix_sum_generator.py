import random
import os
from argparse import ArgumentParser

def is_positive_value(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not > 0")
    return ivalue

def main():
	parser = ArgumentParser(prog='prefix_sum_generator')
	parser.add_argument('N', type=is_positive_value, nargs='+', help="The lengths of the generated lists.");
	parser.add_argument('output_dir', help="The output directory.");
	args = parser.parse_args()

	for n in args.N:
		nums = [0] * n;
		for i in range(n):
			nums[i] = random.randint(-1000, 1000);

		output_file_name = os.path.join(args.output_dir, f"prefix_sum_{n}.init");

		with open(output_file_name, "w") as out_file:
			out_file.writelines([
				"ADL structures 1\n",
				"ListElem Int ListElem ListElem Int\n",
				f"ListElem instances {n} {n}\n"
			]);

			out_file.writelines([f"{val} {idx}\n" for (idx, val) in enumerate(nums)]);

		solution_file_name = os.path.join(args.output_dir, f"prefix_sum_{n}.sol");
		with open(solution_file_name, "w") as out_file:
			prefix_sum = 0;
			for (idx, val) in enumerate(nums):
				prefix_sum += val;
				if (idx + 1) % 5000 == 1:
					out_file.writelines([
						f"({idx+1}, {prefix_sum})\n"
					]);



if __name__ == '__main__':
	main()
