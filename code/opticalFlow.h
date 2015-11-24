#include <opencv2\opencv.hpp>
using namespace cv;

#include "spmbp.h"

class opticalFlow
{
	public:
	Mat im1, im2;
	int height1,width1,height2,width2;
	float m_lambda,m_tau;


	Mat_<Vec2f> flow12, flow21, flow_refined;
	Mat_<uchar> occMap;

	opticalFlow(void);
	~opticalFlow(void);

	void opticalFlowEst(Mat& im1, Mat&im2, Mat_<Vec2f> &flow12, spm_bp_params* params);
	void occMatpEst( Mat_<Vec2f> &flow12, Mat_<Vec2f> &flow21, Mat_<uchar>&occMap);
	void opticalFlowRefine(Mat_<Vec2f> &flow12, Mat_<uchar> &occMap,const Mat_<Vec3b> &weightColorImg, Mat_<Vec2f> &flow_refined);
	void BFsmoothing( const cv::Mat_<cv::Vec2f> &flowVec, const cv::Mat_<cv::Vec3b> &weightColorImg, int radius, cv::Mat_<cv::Vec2f> &refinedFlow );
	void runFLowEstimator(const char* i1file, const char* i2file, const char* seqName, spm_bp_params* params);

	cv::Mat_<cv::Vec3b> currentFlowColorLeft;
	cv::Mat_<cv::Vec3b> currentFlowColorRight;

	void ReadFlowFile(const char *flowFile, cv::Mat_<cv::Vec2f> &flowVec, int height, int width);
	void WriteFlowFile(const char *flowFile, const cv::Mat_<cv::Vec2f> &flowVec, int height, int width);
};