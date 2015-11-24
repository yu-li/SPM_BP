#ifndef COMMON_DATASTRUCTURE_HEADER
#define COMMON_DATASTRUCTURE_HEADER

//#include "Tickcount_Header.h"

#include <vector>
using std::vector;
#include <set>
using std::set;
#include <iterator>
using std::iterator;



typedef struct spm_bp_s {
	//spm-bp
	int max_u; 	//vertical 
	int max_v;	//horizontal	
	int iter_num; //number of iteration
	float alpha; //weight for gradient components
	float tau_c; //truncated value (color)
	float tau_s; //truncated value (smoothness)
	int up_rate; // upsample rate
	float lambda; //smoothness

	//super pixel
	int sp_num;

	//filter kernel
	int kn_size;
	int kn_tau;
	float kn_epsl;

	//display intermediate results
	bool display;
} spm_bp_params;

/* set params to default value */
void spmbp_params_default(spm_bp_params* params);

class GraphStructure
{
public:
	vector<set<int>> adjList;
	int vertexNum;
	GraphStructure();
	GraphStructure(int num);
	~GraphStructure();
	void ReserveSpace(int num);
	void SetVertexNum(int vNum);
	void AddEdge(int s, int e);

	void DeleteEdge(int s, int e);
	void DeleteAllEdge(int s);
};

#endif