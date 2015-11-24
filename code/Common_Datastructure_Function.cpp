#include "Common_Datastructure_Header.h"

void spmbp_params_default(spm_bp_params* params)
{
	params->max_u = 100;
	params->max_v = 150;
	params->iter_num = 5;
	params->alpha = 0.9;
	params->tau_c = 12.0;
	params->tau_s = 2.0;
	params->up_rate = 2;
	params->lambda = 2.0;

	params->sp_num = 500;

	params->kn_size = 9;
	params->kn_tau = 25;
	params->kn_epsl = 1;

	params->display = false;
}



#pragma region GraphStructure_Part

GraphStructure::GraphStructure()
{
	vertexNum = 0;
	adjList.clear();
}

GraphStructure::GraphStructure(int num)
{
	vertexNum = 0;
	adjList.clear();
	ReserveSpace(num);
}

GraphStructure::~GraphStructure()
{

}

void GraphStructure::ReserveSpace(int num)
{
	adjList.reserve(num);
}

void GraphStructure::SetVertexNum(int vNum)
{
	vertexNum = vNum;
	adjList.resize(vertexNum);
}

void GraphStructure::AddEdge(int s, int e)
{
	adjList[s].insert(e);
}

void GraphStructure::DeleteEdge(int s, int e)
{
	adjList[s].erase(adjList[s].find(e));
}

void GraphStructure::DeleteAllEdge(int s)
{
	adjList[s].clear();
}

#pragma endregion