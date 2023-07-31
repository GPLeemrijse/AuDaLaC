// From https://github.com/mCRL2org/mCRL2


// Copyright (c) 2009-2013 University of Twente
// Copyright (c) 2009-2013 Michael Weber <michaelw@cs.utwente.nl>
// Copyright (c) 2009-2013 Maks Verver <maksverver@geocities.com>
// Copyright (c) 2009-2013 Eindhoven University of Technology
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)


/* Adjusted by G.P. Leemrijse */



#include <vector>
#include <assert.h>
#include <chrono>
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
using namespace std::chrono;

typedef std::size_t verti;
typedef std::size_t edgei;
#define NO_VERTEX ((verti)-1)

typedef const verti *const_iterator;

class SCC
{
public:
    SCC(verti V, verti *successors, edgei *successor_index)
        : V(V), nrof_components(0), successors(successors), successor_index(successor_index)
    {
    }

    size_t run()
    {
        // Initialize data structures used in the algorithm
        next_index = 0;
        info.clear();
        info.insert( info.end(), V,
                     std::make_pair(NO_VERTEX, NO_VERTEX) );
        stack.clear();

        // Process all vertices
        for (verti v = 0; v < V; ++v)
        {
            if (info[v].first == NO_VERTEX)
            {
                assert(stack.empty());
                add(v);
                int res = dfs();
                if (res != 0) return res;
            }
        }
        assert(stack.empty());
        return nrof_components;
    }

private:
    void add(verti v)
    {
        // Mark vertex as visited and part of the current component
        info[v].first = info[v].second = next_index++;
        component.push_back(v);

        // Add to stack to be processed in depth-first-search
        stack.push_back(std::make_pair(v, 0));
    }

    /* This implements depth-first-search using a stack, which is a bit more
       complicated but allows us to process arbitrarily large graphs limited
       by available heap space only (instead of being limited by the call
       stack size) as well as conserving some memory. */
    int dfs()
    {
        int res = 0;

        while (res == 0 && !stack.empty())
        {
            verti v = stack.back().first;
            const_iterator edge_it =
                succ_begin(v) + stack.back().second++;

            if (edge_it != succ_end(v))
            {
                // Find next successor `w` of `v`
                verti w = *edge_it;

                if (info[w].first == NO_VERTEX)  // unvisited?
                {
                    add(w);
                }
                else
                if (info[w].second != NO_VERTEX)  // part of current component?
                {
                    /* Check if w's index is lower than v's lowest link, if so,
                       set it to be our lowest link index. */
                    info[v].second = std::min(info[v].second, info[w].first);
                }
            }
            else
            {
                // We're done with this vertex
                stack.pop_back();

                if (!stack.empty())
                {
                    /* Push my lower link index to parent vertex `u`, if it
                       is lower than the parent's current lower link index. */
                    int u = stack.back().first;
                    info[u].second = std::min(info[u].second, info[v].second);
                }

                // Check if v is the component's root (idx == lowest link idx)
                if (info[v].first == info[v].second)
                {
                    // Find v in the current component
                    std::vector<verti>::iterator it = component.end();
                    do {
                        assert(it != component.begin());
                        info[*--it].second = NO_VERTEX;  // mark as removed
                    } while (*it != v);

                    // Call callback functor to handle this component
                    nrof_components++;
                    //res = callback_((const verti*)&*it, component.end() - it);

                    // Remove vertices from current component
                    component.erase(it, component.end());
                }
            }
        }

        return res;
    }

    const_iterator succ_begin(verti v) const {
        return &successors[successor_index[v]];
    }

    const_iterator succ_end(verti v) const {
        return &successors[successor_index[v + 1]];
    }

private:

    verti *successors;
    edgei *successor_index;


    //! nrof vertices
    verti V;

    //! nrof components
    size_t nrof_components;

    //! Index of next vertex to be labelled by inorder traversal.
    verti next_index;

    //! Inorder index and lowest link index of each vertex.
    std::vector<std::pair<verti, verti> > info;

    //! Vertex indices of the current component.
    std::vector<verti> component;

    /*! The depth-first-search stack.

        Each entry consists of a vertex index and an index into its successor
        list.  When a new unvisited vertex `v` is discovered, a pair (`v`, 0)
        is appened at the end of the stack.  The top element is popped off the
        stack when its successor index points to the end of the successor list.
    */
    std::vector< std::pair< verti, verti > > stack;
};


int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Supply a .init file.\n");
        exit(1);
    }

    verti nrof_nodes;
    verti nrof_edges;

    std::ifstream infile(argv[1]);
    std::string line;
    std::string skip;

    while(std::getline(infile, line)){
        if(line.rfind("Edge instances", 0) == 0){
            std::stringstream iss(line);
            iss >> skip >> skip >> skip >> nrof_edges;
            break;
        }
    }

    verti* successors = (verti*)calloc(nrof_edges, sizeof(verti));
    if(successors == NULL) {
        fprintf(stderr, "Could not allocate for successors!\n");
        exit(1);
    }
    std::vector<edgei> succ_index_vec;


    /* Parse edges */
    verti last_s = 0;
    succ_index_vec.push_back(0);

    for (edgei e = 0; e < nrof_edges; e++){
      std::getline(infile, line);
      std::stringstream iss(line);
      verti s_idx;
      verti t_idx;

      iss >> s_idx >> t_idx;
      s_idx--;
      t_idx--;

      for (edgei l = last_s; l < s_idx; l++){
        succ_index_vec.push_back(e);
      }
      last_s = s_idx;
      successors[e] = t_idx;
    }

    std::getline(infile, line);
    if(line.rfind("Node instances", 0) == 0){
        std::stringstream iss(line);
        iss >> skip >> skip >> skip >> nrof_nodes;
    } else {
        fprintf(stderr, "Unexpected line!\n");
        exit(1);
    }

    fprintf(stderr, "Parsing done!\n");

    while(succ_index_vec.size() < nrof_nodes + 1){
        succ_index_vec.push_back(nrof_edges);
    }

    edgei* successor_index = (edgei*)calloc((nrof_nodes+1), sizeof(edgei));
    if(successors == NULL) {
        fprintf(stderr, "Could not allocate for successor_index!\n");
        exit(1);
    }
    std::copy(succ_index_vec.begin(), succ_index_vec.end(), successor_index);
    SCC scc(nrof_nodes, successors, successor_index);
    size_t nrof_components;
    auto t1 = high_resolution_clock::now();
    nrof_components = scc.run();
    auto t2 = high_resolution_clock::now();

    printf("Number of components: %lu\n", nrof_components);

    duration<double, std::milli> ms = t2 - t1;
    std::cerr << "Total walltime CPU: "
            << ms.count()
            << " ms\n";
}