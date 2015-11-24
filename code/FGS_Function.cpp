#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <vector>
using std::vector;

#include "FGS_Header.h"

#define SQ(x) ((x)*(x))

int W, H; // image width, height
int nChannels, nChannels_guide;
double *BLFKernelI; // Kernel LUT

// Main functions
void prepareBLFKernel(double sigma);
void FGS_13(double*** image, double*** joint_image, double sigma, double lambda, int solver_iteration, double solver_attenuation);
void FGS_simple(cv::Mat_<float> &image, cv::Mat_<cv::Vec3f> &joint_image, double sigma, double lambda, int solver_iteration, double solver_attenuation);
void solve_tridiagonal_in_place_destructive(vector<double> &x, const size_t N, vector<double> &a, vector<double> &b, vector<double> &c);

// Memory management
double *** memAllocDouble3(int n,int r,int c);
double** memAllocDouble2(int r,int c);
void memFreeDouble3(double ***p);
void memFreeDouble2(double **p);


// Build LUT for bilateral kernel weight
void prepareBLFKernel(double sigma)
{
	const int MaxSizeOfFilterI = 195075;
	BLFKernelI = (double *)malloc(sizeof(double)*MaxSizeOfFilterI);

	for(int m=0;m<MaxSizeOfFilterI;m++)
		BLFKernelI[m] = exp( -sqrt((double)m)/(sigma) ); // Kernel LUT
}

// mex function call:
// x = mexFGS(input_image, guidance_image = NULL, sigma, lambda, fgs_iteration = 3, fgs_attenuation = 4);
void FGS(const cv::Mat_<float> &in, const cv::Mat_<cv::Vec3b> &color_guide, cv::Mat_<float> &out, double sigma, double lambda, int solver_iteration, int solver_attenuation)
{
	
	// image resolution
	W = in.cols;
	H = in.rows;

    nChannels = 1;
    nChannels_guide = 3;
	
	cv::Mat_<cv::Vec3f> image_guidance;
	color_guide.convertTo(image_guidance,CV_32FC3);

	cv::Mat_<float> image_filtered;
	in.copyTo(image_filtered);

	// run FGS
	sigma *= 255.0;
    
	FGS_simple(image_filtered, image_guidance, sigma, lambda, solver_iteration, solver_attenuation);
	image_filtered.copyTo(out);
}


void FGS_simple(cv::Mat_<float> &image, cv::Mat_<cv::Vec3f> &joint_image, double sigma, double lambda, int solver_iteration, double solver_attenuation)
{
	int color_diff;

	int width = W;
	int height = H;    
	
	vector<double> a_vec(width,0),b_vec(width,0),c_vec(width,0),x_vec(width,0),c_ori_vec(width,0);
	vector<double> a2_vec(width,0),b2_vec(width,0),c2_vec(width,0),x2_vec(width,0),c2_ori_vec(width,0);

	prepareBLFKernel(sigma);
	
	//Variation of lambda (NEW)
	double lambda_in = 1.5*lambda*pow(4.0,solver_iteration-1)/(pow(4.0,solver_iteration)-1.0);
	for(int iter=0;iter<solver_iteration;iter++)
	{
		//for each row
		for(int i=0;i<height;i++)
		{

			for(int j=1;j<width;j++)
			{
                int color_diff = 0;
                // compute bilateral weight for all channels
                for(int c=0;c<nChannels_guide;c++)
                    color_diff += SQ(joint_image[i][j][c] - joint_image[i][j-1][c]);
				
				a_vec[j] = -lambda_in*BLFKernelI[color_diff];		//WLS
			}
			for(int j=0;j<width-1;j++)	c_ori_vec[j] = a_vec[j+1];
			for(int j=0;j<width;j++)	b_vec[j] = 1.f - a_vec[j] - c_ori_vec[j];		//WLS
       
                c_vec = c_ori_vec; 
                for(int j=0;j<width;j++)	x_vec[j] = image[i][j];			
                solve_tridiagonal_in_place_destructive(x_vec, width, a_vec, b_vec, c_vec);
                for(int j=0;j<width;j++)	image[i][j] = x_vec[j];                
		}
		//for each column
		for(int j=0;j<width;j++)
		{
			for(int i=1;i<height;i++)
			{
                int color_diff = 0;
                // compute bilateral weight for all channels
                for(int c=0;c<nChannels_guide;c++)
                    color_diff += SQ(joint_image[i][j][c] - joint_image[i-1][j][c]);

                a2_vec[i] = -lambda_in*BLFKernelI[color_diff];		//WLS
			}
			for(int i=0;i<height-1;i++)
				c2_ori_vec[i] = a2_vec[i+1];
			for(int i=0;i<height;i++)
				b2_vec[i] = 1.f - a2_vec[i] - c2_ori_vec[i];		//WLS
			{
				c2_vec = c2_ori_vec;
                for(int i=0;i<height;i++)	x2_vec[i] = image[i][j];
                solve_tridiagonal_in_place_destructive(x2_vec, height, a2_vec, b2_vec, c2_vec);
                for(int i=0;i<height;i++)	image[i][j] = x2_vec[i];               
            }
		}

		//Variation of lambda (NEW)
		lambda_in /= solver_attenuation;
	}	//iter	
}

void solve_tridiagonal_in_place_destructive(vector<double> &x, const size_t N, vector<double> &a, vector<double> &b, vector<double> &c)
{
	int n;
	
	c[0] = c[0] / b[0];
	x[0] = x[0] / b[0];
	
	// loop from 1 to N - 1 inclusive 
	for (n = 1; n < N; n++) {
		double m = 1.0f / (b[n] - a[n] * c[n - 1]);
		c[n] = c[n] * m;
		x[n] = (x[n] - a[n] * x[n - 1]) * m;
	}
	
	// loop from N - 2 to 0 inclusive 
	for (n = N - 2; n >= 0; n--)
		x[n] = x[n] - c[n] * x[n + 1];
}


double *** memAllocDouble3(int n,int r,int c)
{
	int padding=10;
	double *a,**p,***pp;
	int rc=r*c;
	int i,j;
	a=(double*) malloc(sizeof(double)*(n*rc+padding));
	if(a==NULL) printf("error: out of memory too\n");
	p=(double**) malloc(sizeof(double*)*n*r);
	pp=(double***) malloc(sizeof(double**)*n);
	for(i=0;i<n;i++) 
		for(j=0;j<r;j++) 
			p[i*r+j]=&a[i*rc+j*c];
	for(i=0;i<n;i++) 
		pp[i]=&p[i*r];
	return(pp);
}

void memFreeDouble3(double ***p)
{
	if(p!=NULL)
	{
		free(p[0][0]);
		free(p[0]);
		free(p);
		p=NULL;
	}
}

double** memAllocDouble2(int r,int c)
{
	int padding=10;
	double *a,**p;
	a=(double*) malloc(sizeof(double)*(r*c+padding));
	if(a==NULL) printf("error: out of memory \n");
	p=(double**) malloc(sizeof(double*)*r);
	for(int i=0;i<r;i++) p[i]= &a[i*c];
	return(p);
}
void memFreeDouble2(double **p)
{
	if(p!=NULL)
	{
		free(p[0]);
		free(p);
		p=NULL;
	}
}
