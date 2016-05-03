#include "CF_Filter_Header.h"
//#include "OpenCV_Tickcount_Header.h"

using std::min;
using std::max;


void CFFilter::GetCrossUsingSlidingWindow(const cv::Mat_<cv::Vec3b> &img, cv::Mat_<cv::Vec4b> &crMap, int maxL, int tau)
{
	if ((img.data == NULL) || img.empty()) return;

	////CalcTime ct;
	//ct.Start();

	int width, height;
	width = img.cols;
	height = img.rows;

	crMap.create(height, width);
	int iy, ix;
#pragma omp parallel for private(iy) schedule(guided, 1)
	for (iy=0; iy<height; ++iy)
	{
		// to determine right most, cr[2]
		cv::Vec3i sumColor;
		int lp, rp;
		rp = 0;
		for (lp=0; lp<width; ++lp)
		{
			int diff = rp-lp;
			if (diff == 0)
			{
				sumColor = img[iy][rp];
				++rp;
				++diff;
			}

			while (diff <= maxL)
			{
				if (rp >= width) break;

				//if (abs(sumColor[0]/diff-img[iy][rp][0])>tau || abs(sumColor[1]/diff-img[iy][rp][1])>tau
				//	|| abs(sumColor[2]/diff-img[iy][rp][2])>tau) break;
				if (abs(sumColor[0]-img[iy][rp][0]*diff)>tau*diff 
					|| abs(sumColor[1]-img[iy][rp][1]*diff)>tau*diff
					|| abs(sumColor[2]-img[iy][rp][2]*diff)>tau*diff) break;
				sumColor += img[iy][rp];
				++diff;
				++rp;
			}
			sumColor -= img[iy][lp];

			crMap[iy][lp][2] = diff-1;
		}

		// to determine left most, cr[0]
		lp = width-1;
		for (rp=width-1; rp>=0; --rp)
		{
			int diff = rp-lp;
			if (diff == 0)
			{
				sumColor = img[iy][lp];
				--lp;
				++diff;
			}
			while (diff <= maxL)
			{
				if (lp < 0) break;
				//if (abs(sumColor[0]/diff-img[iy][lp][0])>tau || abs(sumColor[1]/diff-img[iy][lp][1])>tau
				//	|| abs(sumColor[2]/diff-img[iy][lp][2])>tau) break;
				if (abs(sumColor[0]-img[iy][lp][0]*diff)>tau*diff 
					|| abs(sumColor[1]-img[iy][lp][1]*diff)>tau*diff
					|| abs(sumColor[2]-img[iy][lp][2]*diff)>tau*diff) break;
				sumColor += img[iy][lp];
				++diff;
				--lp;
			}

			sumColor -= img[iy][rp];

			crMap[iy][rp][0] = diff-1;
		}
	}

	// determine up and down arm length
#pragma omp parallel for private(ix) schedule(guided, 1)
	for (ix=0; ix<width; ++ix)
	{
		// to determine down most, cr[3]
		cv::Vec3i sumColor;
		int up, dp;
		dp = 0;
		for (up=0; up<height; ++up)
		{
			int diff = dp-up;
			if (diff == 0)
			{
				sumColor = img[dp][ix];
				++dp;
				++diff;
			}
			while (diff <= maxL)
			{
				if (dp >= height) break;
				//if (abs(sumColor[0]/diff-img[dp][ix][0])>tau || abs(sumColor[1]/diff-img[dp][ix][1])>tau
				//	|| abs(sumColor[2]/diff-img[dp][ix][2])>tau) break;
				if (abs(sumColor[0]-img[dp][ix][0]*diff)>tau*diff 
					|| abs(sumColor[1]-img[dp][ix][1]*diff)>tau*diff
					|| abs(sumColor[2]-img[dp][ix][2]*diff)>tau*diff) break;
				sumColor += img[dp][ix];
				++diff;
				++dp;
			}

			sumColor -= img[up][ix];

			crMap[up][ix][3] = diff-1;
		}

		// to determine up most, cr[1]
		up = height-1;
		for (dp=height-1; dp>=0; --dp)
		{
			int diff = dp-up;
			if (diff == 0)
			{
				sumColor = img[up][ix];
				--up;
				++diff;
			}

			while (diff <= maxL)
			{
				if (up < 0) break;
				//if (abs(sumColor[0]/diff-img[up][ix][0])>tau || abs(sumColor[1]/diff-img[up][ix][1])>tau
				//	|| abs(sumColor[2]/diff-img[up][ix][2])>tau) break;
				if (abs(sumColor[0]-img[up][ix][0]*diff)>tau*diff
					|| abs(sumColor[1]-img[up][ix][1]*diff)>tau*diff
					|| abs(sumColor[2]-img[up][ix][2]*diff)>tau*diff) break;
				sumColor += img[up][ix];
				++diff;
				--up;
			}

			sumColor -= img[dp][ix];

			crMap[dp][ix][1] = diff-1;
		}
	}
	
	//ct.End("SlWin Construct CrossMap");
}


void CFFilter::FastCLMF0FloatFilterPointer( const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat_<float> &src, cv::Mat_<float> &dst )
{
//	qx_timer tt;
//	tt.start();

	int iy, ix, width, height;
	width = crMap.cols;
	height = crMap.rows;

	cv::Mat_<float> cost = src;

	// first iteration
	cv::Mat_<float> crossHorSum(height, width);
	cv::Mat_<int> sizeHorSum(height, width);
	cv::Mat_<float> crossHorSumTranspose(width, height);
	cv::Mat_<int> sizeHorSumTranspose(width, height);
	cv::Mat_<cv::Vec4b> crMapTranspose(width, height);
	cv::transpose(crMap, crMapTranspose);

	float *horSum = new float [width+1];
	float *rowSizeHorSum = new float [width+1];
	float *verSum = new float [height+1];
	int *colSizeVerSum = new int [height+1];

	float *costPtr = (float *)(cost.ptr(0));
	float *crossHorPtr = (float *)(crossHorSum.ptr(0));
	int *sizeHorPtr = (int *)(sizeHorSum.ptr(0));
	cv::Vec4b *crMapPtr = (cv::Vec4b *)(crMap.ptr(0));

	for (iy=0; iy<height; ++iy)
	{
		float s = 0.0;		
		float *horPtr = horSum;
		*horPtr++ = s;
		for (ix=0; ix<width; ++ix)
		{
			s += *costPtr++;
			*horPtr++ = s;
		}


		for (ix=0; ix<width; ++ix)
		{
			cv::Vec4b cross = *crMapPtr++;
			*crossHorPtr++ = horSum[ix+cross[2]+1]-horSum[ix-cross[0]];
			*sizeHorPtr++ = cross[2]+cross[0]+1;
		}
	}


	cv::transpose(crossHorSum, crossHorSumTranspose);
	cv::transpose(sizeHorSum, sizeHorSumTranspose);
	crossHorPtr = (float *)(crossHorSumTranspose.ptr(0));
	sizeHorPtr = (int *)(sizeHorSumTranspose.ptr(0));

	cv::Mat_<float> crossVerSum(height, width);
	cv::Mat_<int> sizeVerSum(height, width);
	cv::Mat_<float> crossVerSumTranpose(width, height);
	cv::Mat_<int> sizeVerSumTranpose(width, height);
	float *crossVerPtr = (float *)(crossVerSumTranpose.ptr(0));
	int *sizeVerPtr = (int *)(sizeVerSumTranpose.ptr(0));

	crMapPtr = (cv::Vec4b *)(crMapTranspose.ptr(0));
	const int W_FAC = width;
	for (ix=0; ix<width; ++ix)
	{
		float s = 0.0;
		int cs = 0;	
		

		float *verPtr = verSum;
		int *colSizeVerPtr = colSizeVerSum;
		*verPtr++ = s;
		*colSizeVerPtr++ = cs;
		for (iy=0; iy<height; ++iy)
		{
			s += *crossHorPtr++;
			*verPtr++ =s;
			cs += *sizeHorPtr++;
			*colSizeVerPtr++ = cs;
		}


		for (iy=0; iy<height; ++iy)
		{
			cv::Vec4b cross = *crMapPtr++;
			*crossVerPtr++ = verSum[iy+cross[3]+1]-verSum[iy-cross[1]];
			*sizeVerPtr++ = colSizeVerSum[iy+cross[3]+1]-colSizeVerSum[iy-cross[1]];
		}
	}

	cv::transpose(crossVerSumTranpose, crossVerSum);
	cv::transpose(sizeVerSumTranpose, sizeVerSum);

	// second iteration
	crossVerPtr = (float *)(crossVerSum.ptr(0));
	sizeVerPtr = (int *)(sizeVerSum.ptr(0));
	crossHorPtr = (float *)(crossHorSum.ptr(0));
	sizeHorPtr = (int *)(sizeHorSum.ptr(0));
	crMapPtr = (cv::Vec4b *)(crMap.ptr(0));
	for (iy=0; iy<height; ++iy)
	{
		float s = 0.0;		
		int rs = 0;
		float *horPtr = horSum;
		float *rowSizeHorPtr = rowSizeHorSum;
		*horPtr++ = s;
		*rowSizeHorPtr++ = rs;
		for (ix=0; ix<width; ++ix)
		{
			s += *crossVerPtr++;
			*horPtr++ = s;
			rs += *sizeVerPtr++;
			*rowSizeHorPtr++ = rs;
		}
		
		for (ix=0; ix<width; ++ix)
		{
			cv::Vec4b cross = *crMapPtr++;
			*crossHorPtr++ = horSum[ix+cross[2]+1]-horSum[ix-cross[0]];
			*sizeHorPtr++ = rowSizeHorSum[ix+cross[2]+1]-rowSizeHorSum[ix-cross[0]];
		}
	}
	cv::transpose(crossHorSum, crossHorSumTranspose);
	cv::transpose(sizeHorSum, sizeHorSumTranspose);
	crossHorPtr = (float *)(crossHorSumTranspose.ptr(0));
	sizeHorPtr = (int *)(sizeHorSumTranspose.ptr(0));
	crossVerPtr = (float *)(crossVerSumTranpose.ptr(0));
	sizeVerPtr = (int *)(sizeVerSumTranpose.ptr(0));
	crMapPtr = (cv::Vec4b *)(crMapTranspose.ptr(0));
	for (ix=0; ix<width; ++ix)
	{
		float s = 0.0;
		int cs = 0;
		float *verPtr = verSum;
		int *colSizeVerPtr = colSizeVerSum;
		*verPtr++ = s;
		*colSizeVerPtr++ = cs;

		for (iy=0; iy<height; ++iy)
		{
			s += *crossHorPtr++;
			*verPtr++ = s;
			cs += *sizeHorPtr++;
			*colSizeVerPtr++ = cs;
		}

		for (iy=0; iy<height; ++iy)
		{
			cv::Vec4b cross = *crMapPtr++;
			*crossVerPtr++ = verSum[iy+cross[3]+1]-verSum[iy-cross[1]];
			*sizeVerPtr++ = colSizeVerSum[iy+cross[3]+1]-colSizeVerSum[iy-cross[1]];
		}

	}
	//tt.time_display("Smoothing");

	delete [] horSum;
	delete [] rowSizeHorSum;
	delete [] verSum;
	delete [] colSizeVerSum;

	cv::transpose(crossVerSumTranpose, crossVerSum);
	cv::transpose(sizeVerSumTranpose, sizeVerSum);

	dst.create(height, width);
	cv::Mat_<float> tmpDst = dst;
	float *pCrossVerSum = (float *)crossVerSum.ptr(0);
	int *pSizeVerSum = (int *)sizeVerSum.ptr(0);

	float *pTmpDst = (float *)tmpDst.ptr(0);

	for (iy=0; iy<height*width; ++iy)
	{
		(*pTmpDst++) = (*pCrossVerSum++)/(*pSizeVerSum++);
	}
}


