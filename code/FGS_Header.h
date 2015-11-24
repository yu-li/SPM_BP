#include "opencv2/opencv.hpp"


// x = FGS(input_image, guidance_image = NULL, sigma, lambda, fgs_iteration = 3, fgs_attenuation = 4);
void FGS(const cv::Mat_<float> &in, const cv::Mat_<cv::Vec3b> &color, cv::Mat_<float> &out, double sigma, double lambda, int fgs_iteration = 3, int fgs_attenuation = 4);
