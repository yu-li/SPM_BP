#ifndef MOTION_COLOR_HEADER
#define MOTION_COLOR_HEADER
#include "opencv2/opencv.hpp"


void computeColor(float fx, float fy, uchar *pix);
void MotionToColor(const cv::Mat &motion, cv::Mat &colorMat, float maxmotion);
#endif