#ifndef CF_FILTER_HEADER
#define CF_FILTER_HEADER

#include "opencv2/opencv.hpp"

#include <omp.h>

#include <deque>
using std::deque;

#include <string>
using std::string;

#include <cstdio>
#include <algorithm>

class CFFilter
{
public:
	cv::Mat_<cv::Vec3b> img;
	cv::Mat_<cv::Vec4b> crossMapN;
	cv::Mat_<cv::Vec4b> crossMapSW;

	cv::Mat_<cv::Vec4b> crossGrad;

	CFFilter(){};
	~CFFilter(){};

	// build support cross
	void GetCrossUsingSlidingWindow(const cv::Mat_<cv::Vec3b> &img, cv::Mat_<cv::Vec4b> &crMap, int maxL, int tau);

	void EnsureSmallestSupportArm(const cv::Mat_<cv::Vec4b> &crMap, int minArm, cv::Mat_<cv::Vec4b> &crMapOut);

	// some cross related function for test
	void ExpandCrossToRegionFromSeed(const cv::Mat_<cv::Vec4b> &crMap, cv::Point seed, cv::Mat &labelMask);

	void CountCoverdCross(const cv::Mat_<cv::Vec4b> &crMap, cv::Mat &_ccNum, const char* winName);
	void CountCoverdCrossRegion(const cv::Mat_<cv::Vec4b> &crMap, cv::Mat &_ccNum, const char* winName);

	void NormalizeFloatAndShow( const cv::Mat &flMat, const char* winName );
	void NormalizeUshortAndShow(const cv::Mat &usMat, const char* winName);
	void NormalizeDoubleAndShow(const cv::Mat &doMat, const char* winName);

	void DrawCrossArmPlot(const cv::Mat_<cv::Vec4b> &crMap, int row, int col);

	// filter functions
	// summation on color cv::vec3b type
	void SummationColorOnCrossMapRowMajored(const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat &src, cv::Mat &dst);
	// to be done
	void SummationColorOnCrossMapColMajored(const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat &src, cv::Mat &dst);

	void SummationColorOnRect(const cv::Mat &src, cv::Mat &dst, int rectSize);

	// summation on double type data
	//void SummationDoubleOnCrossMap(const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat &src, cv::Mat &dst);
	void SummationDoubleOnCrossMapColMajored(const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat &src, cv::Mat &dst);
	void SummationDoubleOnCrossMapRowMajored(const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat &src, cv::Mat &dst);
	void SummationDoubleOnRect(const cv::Mat &src, cv::Mat &dst, int rectSize);

	void Summation3CDoubleOnCrossMapColMajored(const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat &src, cv::Mat &dst);

	// count cross support region size; count is ushort
	void CountSizeOnCrossMapColMajored(const cv::Mat_<cv::Vec4b> &crMap, cv::Mat &count);
	void CountSizeOnCrossMapRowMajored(const cv::Mat_<cv::Vec4b> &crMap, cv::Mat &count);

	// count rect region size, crMap is used to pass dimensions
	void CountSizeOnRect(const cv::Mat_<cv::Vec4b> &crMap, cv::Mat &count, int rectSize);

public:
	void AverageImageUsingCrossMap(const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat &src, cv::Mat &dst);
	void CalacuteCrossRectDifference(const cv::Mat &img, int maxL, int tau, cv::Mat &diffRes);

	// cross-based filter, parameter L, tau, lambda
	void FilterColorImageUsingCrossMapRowMajored(const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat &src, cv::Mat &dst, int maxL, int tau, double lambda);
	void FilterColorImageUsingCrossMapColMajored(const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat &src, cv::Mat &dst, int maxL, int tau, double lambda);
	void FilterColorImageUsingCrossMapRowAndCol(const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat &src, cv::Mat &dst, int maxL, int tau, double lambda);

	// cost volume, double type, parameter L, tau, lambda
	void FilterCostVolumeByCrossMapColMajored(const cv::Mat &costVolume, const cv::Mat_<cv::Vec4b> &crMap, cv::Mat &filteredVolume, int maxL, int tau, double lambda);
	void FilterCostVolumeByCrossMapRowMajored(const cv::Mat &costVolume, const cv::Mat_<cv::Vec4b> &crMap, cv::Mat &filteredVolume, int maxL, int tau, double lambda);
	void FilterCostVolumeByCrossMapRowAndCol(const cv::Mat &costVolume, const cv::Mat_<cv::Vec4b> &crMap, cv::Mat &filteredVolume, int maxL, int tau, double lambda);

	// test in LUV space, cv::Vec3f type
	void GetCrossUsingSlidingWindowInLUV(const cv::Mat_<cv::Vec3f> &img, cv::Mat_<cv::Vec4b> &crMap, int maxL, int tau);
	// filter in LUV space

	// CLMF0
	void CLMFilter0ColorImageColMajored(const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat &src, cv::Mat &dst, int maxL, int tau);
	void CLMFilter0CostVolumeColMajored(const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat &src, cv::Mat &dst, int maxL, int tau);

	// weighted on original image; give weight of original pixel based on its support cross size
	void WeightedFilterColorImageColMajored(const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat &src, cv::Mat &dst, int maxL, int tau);

	// [added], gradient of image based on arm support
	void CalculateGradientBasedOnCrossmap(const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat &img);
	void FastCrossBasedFilter(const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat &src, cv::Mat &dst);

	// [added] for kinecut
	void GetCrossUsingSlidingWindowWithDepth(const cv::Mat_<cv::Vec3b> &img, const cv::Mat &depth, cv::Mat_<cv::Vec4b> &crMap, int maxL, int tau);

	// [added] for fast filter
	void GetCrossGradientUsingSlidingWindow(const cv::Mat_<cv::Vec3b> &img, cv::Mat_<cv::Vec4b> &crGrad, int maxL, int tau);
	void FastDoubleFilterUsingCrossGradient(const cv::Mat_<cv::Vec4b> &crGrad, const cv::Mat &src, cv::Mat &dst);

	// [added] for fast CLMF0 filter
	void FastCLMF0DoubleFilter(const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat_<double> &src, cv::Mat_<double> &dst);
	void FastCLMF0FloatFilter( const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat_<float> &src, cv::Mat_<float> &dst );

	// [added] a wrapper for skelton build
	void WrapperForSkelonBuild(const cv::Mat_<cv::Vec3b> &img, int armLength, cv::Mat_<cv::Vec4b> &crMap);

	// [added] 
	void FastCLMF0FloatFilterPointer( const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat_<float> &src, cv::Mat_<float> &dst );
};

#endif