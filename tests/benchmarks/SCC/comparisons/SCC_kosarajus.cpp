//  Kosaraju’s algorithm to find strongly connected components of a directed graph
// Given a directed graph, find all the strongly connected components

#include <iostream>
#include <cstdio>
#include <vector>
#include <stack>
#include <chrono>


using namespace std;
using namespace std::chrono;


vector<vector<uint>> graph;              //store graph's adjacency list
vector<vector<uint>> transpose_graph;    //store transpose graph's adjacency list
vector<bool> vis;                       //store node visit stat for dfs

vector<uint> node_order;     //store nodes in order of their finish time in dfs

//this function travels the whole graph and puts the nodes in a node_order
// the nodes are pushed in order of their finish time
void dfs(uint n){
    if(vis[n]) return;  //if node is already visited
    
    uint len= graph[n].size();
    vis[n]= true;
    
    for(uint i= 0; i<len; i++){
        dfs(graph[n][i]);
    }
    
    node_order.push_back(n);
}


//this function traverses the transpose graph
// at each call traverses a SCC and prints each node of that SCC to std output
void dfs_print(uint n){
    if(vis[n] == true) return;  //if node is already visited
    
    cout<< n <<" ";
    
    uint len= transpose_graph[n].size();
    vis[n]= true;
    
    for(uint i= 0; i<len; i++){
        dfs_print(transpose_graph[n][i]);
    }
}

//n nodes
// print each SCC nodes in separate line
// and return the number of SCC in that graph
int kosarajuSCC(uint n){
    uint scc_count= 0;   //keep count of strongly connected components
    for(uint i= 0; i<n; i++){
        if(vis[i] == false){
            dfs(i);
        }
    }
    
    for(uint i= 0; i<n; i++){
        vis[i]= false;
    }
    
    for(int i= node_order.size()-1; i>= 0; i--){
        if(vis[node_order[i]] == false){
            dfs_print(node_order[i]);
            scc_count++;
            cout<<endl;
        }
        
    }
    
    node_order.clear();
    return scc_count;
}

int main(void){
    
    uint n; //n nodes
    uint m; //m edges
    uint u; //start node of an edge
    uint v; //end node of an edge
    
    cin>>n>>m;
    
    vis.reserve(n);
    graph.reserve(n);
    transpose_graph.reserve(n);

    
    //initialize
    for(uint i= 0; i< n; i++){
        vector<uint> g;
        vector<uint> tg;

        vis.push_back(false);
        graph.push_back(g);
        transpose_graph.push_back(tg);
    }
    
    //take graph input as adjacency list
    for(uint i= 0; i<m; i++){
        cin>>u>>v;     //edge u -> v
        
        graph[u].push_back(v);
        transpose_graph[v].push_back(u);
    }

    auto t1 = high_resolution_clock::now();
    int components= kosarajuSCC(n);
    auto t2 = high_resolution_clock::now();
    cout<< "Components: "<< components <<endl;

    duration<double, std::milli> ms = t2 - t1;

    std::cout << "Total walltime CPP "
              << ms.count()
              << " ms\n";
    
    
    return 0;
}