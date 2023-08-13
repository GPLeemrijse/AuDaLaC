#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <assert.h>
#include <chrono>
using namespace std::chrono;

int main(int argc, char **argv) {
	if (argc != 2) {
		printf("Supply a .init file.\n");
		exit(1);
	}
	std::ifstream infile(argv[1]);
	std::string line;
	bool parsing = false;
	std::vector<int> numbers;
	while(std::getline(infile, line)){
		if(parsing) {
			int num;
			std::istringstream iss(line);
			iss >> num;
			numbers.push_back(num);
		} else if(line.rfind("ListElem Int ListElem ListElem Int", 0) == 0) {
			int nrof_elements1;
			int nrof_elements2;
			std::string listelems_str;
			std::string instances_str;
			infile >> listelems_str >> instances_str >> nrof_elements1 >> nrof_elements2;
			assert(nrof_elements1 == nrof_elements2);
			numbers.reserve(nrof_elements1);
			std::getline(infile, line);
			parsing = true;
		}	
	}

	std::vector<int> sums;
	int sum = 0;
	sums.reserve(numbers.size());
	auto t1 = high_resolution_clock::now();
	for(int n : numbers){
		sum += n;
		sums.push_back(sum);
	}
	auto t2 = high_resolution_clock::now();
	std::cout << "Sum: " << sum << std::endl;

	for(int i = 0; i < sums.size(); i += 5000){
		printf("prefix %u: %d\n", i, sums[i]);
	}
	
	duration<double, std::milli> ms = t2 - t1;

    std::cerr << "Total walltime CPU: "
              << ms.count()
              << " ms\n";
}