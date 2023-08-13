#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <assert.h>
#include <chrono>
#include <algorithm> 
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
			if(line.rfind("Printer instances", 0) == 0) {
				break;
			}
			int num;
			std::istringstream iss(line);
			iss >> num;
			numbers.push_back(num);
		} else if(line.rfind("Printer ListElem", 0) == 0) {
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
	
	auto t1 = high_resolution_clock::now();
	std::sort(numbers.begin(), numbers.end());
	auto t2 = high_resolution_clock::now();
	
	printf("numbers: %d, %d,... %d, %d\n", numbers[0], numbers[1], numbers[numbers.size()-2], numbers[numbers.size()-1]);

	duration<double, std::milli> ms = t2 - t1;

    std::cerr << "Total walltime CPU: "
              << ms.count()
              << " ms\n";
}