#include <iostream>
#include <string>
#include "stdlib.h"
#include <vector>


#include <opencv2\opencv.hpp>
#include <iostream>
#include <string>
#include "stdlib.h"
#include <vector>

using namespace cv;
using namespace std;

#include "opticalFlow.h"
#include "omp.h"


/* show help information */
void help(){
    printf("USAGE: spm-bp image1 image2 outputfile [options]\n");
	printf("\n");
    printf("Estimate optical flow field between two iamges with SPM_BP and store it into a .flo file\n");
    printf("\n");
    printf("options:\n"); 
    printf("spm-bp parameters\n");
	printf(" -it_num	<int>(5)	number of iterations\n"); 
	printf(" -sp_num	<int>(500)	number of superpixels\n");
	printf(" -max_u		<int>(100)	motion range in pixels(ver)\n"); 
	printf(" -max_v		<int>(200)	motion range in pixels(hor)\n"); 
	printf(" -kn_size	<int>(9)	filter kerbel radius\n");
	printf(" -kn_tau	<int>(25)	filter smootheness\n");
	printf(" -lambda	<float>(2)	pairwise smoothness\n");
    printf("\n");
}

int main(int argc, char **argv){
	if(argc<4)	{help(); exit(1);}

	// read filenames
	const char *frame1file = argv[1];
	const char *frame2file = argv[2];
	const char *outputfile = argv[3];
	
	spm_bp_params params;
    spmbp_params_default(&params);
	// read optional arguments 
    #define isarg(key)  !strcmp(a,key)
    int current_arg = 4;
    while(current_arg < argc ){
        const char* a = argv[current_arg++];
        if( isarg("-max_u") ) 
            params.max_u = atoi(argv[current_arg++]);
        else if( isarg("-max_v") ) 
            params.max_v = atoi(argv[current_arg++]); 
		else if( isarg("-sp_num") ) 
            params.sp_num = atoi(argv[current_arg++]);
        else if( isarg("-it_num") ) 
            params.iter_num = atoi(argv[current_arg++]);
		else if( isarg("-kn_size") ) 
            params.kn_size = atoi(argv[current_arg++]); 
		else if( isarg("-kn_tau") ) 
            params.kn_tau = atoi(argv[current_arg++]);
        else if( isarg("-lambda") ) 
            params.lambda = atof(argv[current_arg++]);
		else if( isarg("-verbose") ) 
            params.display = true;
		else{
            fprintf(stderr, "unknown argument %s", a);
            help();
            exit(1);
        }
	}
	opticalFlow of_est;
	of_est.runFLowEstimator(frame1file, frame2file, outputfile, &params);
}