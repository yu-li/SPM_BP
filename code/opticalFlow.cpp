
#include "opticalFlow.h"

#include "FlowInputOutput.h"
#include "colorcode.h"

#include "FGS_Header.h"

opticalFlow::opticalFlow(void){
}

opticalFlow::~opticalFlow(void){
	flow12.release();
	flow21.release();
	flow_refined.release();
}

void opticalFlow::runFlowEstimator(const char* i1file, const char* i2file, const char* seqName, spm_bp_params* params)
{
	Mat in1 = imread(i1file);
	Mat in2 = imread(i2file);

	cv::GaussianBlur(in1,in1,cv::Size(0,0),0.9);
	cv::GaussianBlur(in2,in2,cv::Size(0,0),0.9);
	in1.copyTo(im1);
	in2.copyTo(im2);

	height1 = im1.rows;	width1 = im1.cols;
	height2 = im2.rows;	width2 = im2.cols;
	
	flow12.create(height1,width1); 
	flow21.create(height2,width2);
	occMap.create(height1,width1);

	
#pragma omp parallel for
    for(int i = 0 ; i<2 ; ++i)
    {
        if(i==0)
	//optical flow from frame 1 to frame 2	
	{opticalFlowEst( im1, im2, flow12, params);}
	
	else
	//optical flow from frame 2 to frame 1
	{opticalFlowEst( im2, im1, flow21, params);} 
   }
	//left-right consistancy check (occlusion estimation)
	occMap.create(height1,width1);
	occMapEst( flow12, flow21, occMap);

	//post processing (occlusion region filling)
	opticalFlowRefine(flow12, occMap, im1, flow_refined);

	//write result
	Mat_<Vec3b> flow_color_t;
	MotionToColor(flow_refined, flow_color_t, -1);
	// cv::imshow("Flow aft Post-processing",flow_color_t);
	cv::imwrite("flow_visualization.png",flow_color_t);
	flow_color_t.release();
	char flow_file_name[200];
	sprintf(flow_file_name,"%s.flo",seqName);
	WriteFlowFile(flow_file_name,flow_refined,height1,width1);
}

void opticalFlow::opticalFlowEst(Mat& img1, Mat&img2, Mat_<Vec2f> &flow, spm_bp_params* params)
{
	spm_bp *estimator = new spm_bp();
	estimator->loadPairs(img1,img2);

	// load params
	estimator->setParameters(params);
	
	// preparare data
	estimator->preProcessing();

	// spm-bp main start
	cv::Mat_<cv::Vec2f> flowResult;
	flowResult.create(height1,width1);
	estimator->runspm_bp(flowResult);

	flow = flowResult.clone();
	delete estimator;
}

void opticalFlow::occMapEst( Mat_<Vec2f> &flow12, Mat_<Vec2f> &flow21, Mat_<uchar>&occMap)
{
	int iy, ix;

	const float FLOW_PIXEL_THRESH = 2;

	occMap.setTo(255);
	for (iy=0; iy<height1; ++iy)
	{
		for (ix=0; ix<width1; ++ix)
		{
			Vec2f fFlow = flow12[iy][ix];
			int ny, nx;
			ny = floor(iy+fFlow[1]+0.5);
			nx = floor(ix+fFlow[0]+0.5);

			if (ny>=0 && ny<height1 && nx>=0 && nx<width1) 
			{
				cv::Vec2f bFlow = flow21[ny][nx];
				if (fabs(bFlow[1]+ny-iy)<FLOW_PIXEL_THRESH && fabs(bFlow[0]+nx-ix)<FLOW_PIXEL_THRESH)
				{
					continue;
				}
			}
			occMap[iy][ix] = 0;
		}
	}

	Mat bw = occMap;
    Mat labelImage(occMap.size(), CV_32S);
    int nLabels = connectedComponents(bw, labelImage, 8);

	vector<int> hist(nLabels,0);
	for (iy=0; iy<height1; ++iy)
		for (ix=0; ix<width1; ++ix)
			hist[labelImage.at<int>(iy,ix)]++;
	vector<int> rmv_list;
	rmv_list.reserve(20);
	for (int i=0;i<nLabels;++i){
		if (hist[i]<50)
			rmv_list.push_back(i);
	}
	for (iy=0; iy<height1; ++iy)
	{
		for (ix=0; ix<width1; ++ix)
		{
			for (int r=0; r<rmv_list.size(); ++r)
			if(labelImage.at<int>(iy,ix) == rmv_list[r])
				occMap[iy][ix] = 0;
		}
	}
}


void opticalFlow::opticalFlowRefine(Mat_<Vec2f> &flow_in, Mat_<uchar> &occMap,const Mat_<Vec3b> &weightColorImg, Mat_<Vec2f> &flow_refined)
{
	Mat_<float> flow_in_single[2];
	split(flow_in,flow_in_single);
	Mat_<float> flow_out_single[2];
	Mat_<float> occ_fgs;
	occMap.convertTo(occ_fgs,CV_32FC1);
	occMap = occMap;
	multiply(flow_in_single[0],occ_fgs,flow_in_single[0]);
	multiply(flow_in_single[1],occ_fgs,flow_in_single[1]);
	FGS(flow_in_single[0],weightColorImg,flow_out_single[0], 0.01, 100);
	FGS(flow_in_single[1],weightColorImg,flow_out_single[1], 0.01, 100);
	FGS(occMap,weightColorImg,occ_fgs, 0.01, 100);
	divide(flow_out_single[0],occ_fgs,flow_out_single[0]);
	divide(flow_out_single[1],occ_fgs,flow_out_single[1]);

	merge(flow_out_single,2,flow_refined);
}

void opticalFlow::ReadFlowFile( const char *flowFile, cv::Mat_<cv::Vec2f> &flowVec, int height, int width )
{
	float *fBuffer = new float[height*width*2];
	::ReadFlowFile(fBuffer, flowFile, height, width);
	int iy, ix;
	flowVec.create(height, width);
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			float tmp0, tmp1;
			tmp0 = fBuffer[iy*2*width+2*ix];
			tmp1 = fBuffer[iy*2*width+2*ix+1];
			flowVec[iy][ix][0] = tmp0;
			flowVec[iy][ix][1] = tmp1;
		}
	}

	delete [] fBuffer;
}


void opticalFlow::WriteFlowFile( const char *flowFile, const cv::Mat_<cv::Vec2f> &flowVec, int height, int width )
{
	float *fBuffer = new float[height*width*2];
	int iy, ix;
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			fBuffer[iy*2*width+2*ix] = flowVec[iy][ix][0];
			fBuffer[iy*2*width+2*ix+1] = flowVec[iy][ix][1];
		}
	}

	::WriteFlowFile(fBuffer, flowFile, height, width);
	delete [] fBuffer;
}




