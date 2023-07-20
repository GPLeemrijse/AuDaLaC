#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <assert.h>
#include <chrono>
#include <algorithm> 
using namespace std::chrono;


typedef struct {
	bool X_m;
	bool X_0;
	bool X_k;
	bool N;
	bool B;
	bool Y;

	void init_N(void){
		N = X_m && X_k;
	}

	void init_B(void){
		B = X_k && !N;
	}

	void remove_bad(bool* stable){
		if(X_k != (X_k && !B)){
			X_k = X_k && !B;
			*stable = false;
		}
	}

	void init_Y(void){
		Y = X_0 && X_k;
	}

	void print(int state_num){
		printf("%04d: X_m: %d, X_0: %d, X_k: %d, N: %d, B: %d, Y: %d\n",
			state_num,
			(int)X_m,
			(int)X_0,
			(int)X_k,
			(int)N,
			(int)B,
			(int)Y
		);
	}
} state_t;

int main(int argc, char **argv) {
	if (argc != 2) {
		printf("Supply a .init file.\n");
		exit(1);
	}

	int nrof_states;
	std::vector<state_t> states;
	int nrof_controllable;
	std::vector<std::pair<state_t*, state_t*>> controllable;
	std::vector<std::pair<uint, uint>> controllable_idxs;
	int nrof_uncontrollable;
	std::vector<std::pair<state_t*, state_t*>> uncontrollable;

	std::ifstream infile(argv[1]);
	std::string line;
	
	/* Parse until nrof states controllable */
	while(std::getline(infile, line)){
		if(line.rfind("UncontrollableEvent State State", 0) == 0){
			break;
		}
	}

	/* Parse controllable */
  std::string skip;
	infile >> skip >> skip >> skip >> nrof_controllable;
	controllable.reserve(nrof_controllable);
	controllable_idxs.reserve(nrof_controllable);
	std::getline(infile, line);
	for (int c = 0; c < nrof_controllable; c++){
      std::getline(infile, line);
      std::stringstream iss(line);
      uint s_idx;
      uint t_idx;

      iss >> s_idx
      		>> t_idx;
      
      controllable_idxs.push_back(std::pair(s_idx-1, t_idx-1));
  }

	/* Parse states */
	infile >> skip >> skip >> skip >> nrof_states;
	states.reserve(nrof_states);
	std::getline(infile, line);
  for (int s = 0; s < nrof_states; s++){
      std::getline(infile, line);
      std::stringstream iss(line);
      states.push_back({0});
      
      iss >> states.back().X_m
      		>> states.back().X_0;

      states.back().X_k = true;
      states.back().N = false;
      states.back().B = false;
      states.back().Y = false;
  }

  for(auto &[s, t] : controllable_idxs){
  	controllable.push_back(std::pair(&states[s], &states[t]));
  }

  /* Parse uncontrollable */
	infile >> skip >> skip >> skip >> nrof_uncontrollable;
	uncontrollable.reserve(nrof_uncontrollable);
	std::getline(infile, line);
	for (int c = 0; c < nrof_uncontrollable; c++){
      std::getline(infile, line);
      std::stringstream iss(line);
      uint s_idx;
      uint t_idx;

      iss >> s_idx
      		>> t_idx;
      
      uncontrollable.push_back(std::pair(&states[s_idx-1], &states[t_idx-1]));
  }

  printf("Nrof states: %u, nrof_controllable: %u, nrof_uncontrollable: %u\n", nrof_states, nrof_controllable, nrof_uncontrollable);
  
	auto t1 = high_resolution_clock::now();
	bool X_k_stable = true;
	do {
		X_k_stable = true;

		// init_N
		for(auto &s : states) {s.init_N();}
		
		// Fix(compute_nonblocking)
		bool N_k_stable = true;
		do {
			N_k_stable = true;

			// compute_nonblocking
			for (auto &[x, y] : controllable) {
				if(x->X_k && y->N && !x->N){
					x->N = true;
					N_k_stable = false;
				}
			}
		} while(!N_k_stable);

		// init_B
		for(auto &s : states) {s.init_B();}

		// Fix(compute_bad)
		bool B_k_stable = true;
		do {
			B_k_stable = true;

			// compute_bad
			for (auto &[x, y] : uncontrollable) {
				if(x->X_k && y->B && !x->B){
					x->B = true;
					B_k_stable = false;
				}
			}
		} while(!B_k_stable);

		// Remove bad
		for(auto &s : states) {s.remove_bad(&X_k_stable);}
	} while(!X_k_stable);
	
	// init Y
	for(auto &s : states) {s.init_Y();}

	// Fix(compute_supervisor)
	bool Y_stable = true;
	do {
		Y_stable = true;

		// compute_supervisor
		for (auto &[x, y] : controllable) {
			if(y->X_k && x->Y && !y->Y){
				y->Y = true;
				Y_stable = false;
			}
		}
	} while(!Y_stable);

	auto t2 = high_resolution_clock::now();

	int nrof_y_states = 0;
	for(auto &s : states){
		if(s.Y){
			nrof_y_states++;
		}
	}

	int nrof_b_states = 0;
	for(auto &s : states){
		if(s.B){
			nrof_b_states++;
		}
	}

	printf("Number of Y states: %d\n", nrof_y_states);
	printf("Number of B states: %d\n", nrof_b_states);

	duration<double, std::milli> ms = t2 - t1;
  std::cerr << "Total walltime CPU: "
            << ms.count()
            << " ms\n";
}