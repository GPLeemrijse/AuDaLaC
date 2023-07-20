#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <assert.h>
#include <chrono>
#include <algorithm> 
using namespace std::chrono;


class State {
public:
	bool marked;
	bool initial;
	bool deleted;
	bool N;
	bool B;
	bool Y;

	std::vector<State*> c_pred;
	std::vector<State*> c_succ;
	std::vector<State*> u_pred;

	State(bool marked, bool initial) : 
		marked(marked), initial(initial),
		deleted(false), N(false), B(false), Y(false) {}


	void init_nonblocking(void){
		N = !deleted && marked;
	}

	void start_nonblocking(void){
		if(N) {
			propagate_nonblocking();
		}
	}

	void propagate_nonblocking(void){
		N = true;
		for(State* cp : c_pred){
			if(!cp->N && !cp->deleted){
				cp->propagate_nonblocking();
			}
		}
	}

	void init_bad(void){
		B = !deleted && !N;
	}

	void start_bad(void){
		if(B) {
			propagate_bad();
		}
	}

	void propagate_bad(void){
		B = true;
		for(State* cp : u_pred){
			if(!cp->B && !cp->deleted){
				cp->propagate_bad();
			}
		}
	}

	void init_supervisor(void){
		Y = !deleted && initial;
	}

	void start_supervisor(void){
		if(Y) {
			propagate_supervisor();
		}
	}

	void propagate_supervisor(void){
		Y = true;
		for(State* cp : c_succ){
			if(!cp->Y && !cp->deleted){
				cp->propagate_supervisor();
			}
		}
	}

	void delete_bad(bool* stable){
		if(B && !deleted) {
			deleted = true;
			*stable = false;
		}
	}

};

int main(int argc, char **argv) {
	if (argc != 2) {
		printf("Supply a .init file.\n");
		exit(1);
	}

	int nrof_states;
	std::vector<State> states;
	int nrof_controllable;
	std::vector<std::pair<uint, uint>> controllable;
	int nrof_uncontrollable;
	std::vector<std::pair<uint, uint>> uncontrollable;

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
	std::getline(infile, line);
	for (int c = 0; c < nrof_controllable; c++){
      std::getline(infile, line);
      std::stringstream iss(line);
      uint s_idx;
      uint t_idx;

      iss >> s_idx
      		>> t_idx;
      
      controllable.push_back(std::pair(s_idx-1, t_idx-1));
  }

	/* Parse states */
	infile >> skip >> skip >> skip >> nrof_states;
	states.reserve(nrof_states);
	std::getline(infile, line);
  for (int s = 0; s < nrof_states; s++){
      std::getline(infile, line);
      std::stringstream iss(line);
      bool marked;
      bool initial;

      iss >> marked
      		>> initial;

      states.push_back(State(marked, initial));
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
      
      uncontrollable.push_back(std::pair(s_idx-1, t_idx-1));
  }

  for(auto &[s, t] : controllable){
  	states[s].c_succ.push_back(&states[t]);
  	states[t].c_pred.push_back(&states[s]);
  }

  for(auto &[s, t] : uncontrollable){
  	states[t].u_pred.push_back(&states[s]);
  }

  printf("Nrof states: %u, nrof_controllable: %u, nrof_uncontrollable: %u\n", nrof_states, nrof_controllable, nrof_uncontrollable);
  
	auto t1 = high_resolution_clock::now();
	bool no_deletions = true;
	do {
		no_deletions = true;
		
		// Computes non-blocking
		for(auto &s : states) {s.init_nonblocking();}
		for(auto &s : states) {s.start_nonblocking();}

		// Computes bad
		for(auto &s : states) {s.init_bad();}
		for(auto &s : states) {s.start_bad();}
		
		// Remove bad
		for(auto &s : states) {s.delete_bad(&no_deletions);}
	} while(!no_deletions);
	
	// Computes supervisor
	for(auto &s : states) {s.init_supervisor();}
	for(auto &s : states) {s.start_supervisor();}

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