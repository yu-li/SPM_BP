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

void CFFilter::GetCrossUsingSlidingWindowWithDepth(const cv::Mat_<cv::Vec3b> &img, const cv::Mat &depth, cv::Mat_<cv::Vec4b> &crMap, int maxL, int tau)
{
	if ((img.data == NULL) || img.empty()) return;

	////CalcTime ct;
	//ct.Start();

	int width, height;
	width = img.cols;
	height = img.rows;

	cv::Mat_<ushort> dep = depth;
	const ushort DEPTH_DIFF_LIMIT = 200;

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
				// added using depth
				// if ((dep[iy][rp]!=0) && (dep[iy][lp]!=0) && (abs(dep[iy][rp]-dep[iy][lp])>DEPTH_DIFF_LIMIT)) break;
				if (abs(dep[iy][rp]-dep[iy][lp])>DEPTH_DIFF_LIMIT) break;

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

				// added using depth
				// if ((dep[iy][rp]!=0) && (dep[iy][lp]!=0) && (abs(dep[iy][rp]-dep[iy][lp])>DEPTH_DIFF_LIMIT)) break;
				if (abs(dep[iy][rp]-dep[iy][lp])>DEPTH_DIFF_LIMIT) break;

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
				//if (abs(sumColor[0]/diff-img[dp][p.x][0])>tau || abs(sumColor[1]/diff-img[dp][ix][1])>tau
				//	|| abs(sumColor[2]/diff-img[dp][ix][2])>tau) break;
				if (abs(sumColor[0]-img[dp][ix][0]*diff)>tau*diff 
					|| abs(sumColor[1]-img[dp][ix][1]*diff)>tau*diff
					|| abs(sumColor[2]-img[dp][ix][2]*diff)>tau*diff) break;

				// added using depth
				// if ((dep[dp][ix]!=0) && (dep[up][ix]!=0) && (abs(dep[dp][ix]-dep[up][ix])>DEPTH_DIFF_LIMIT)) break;
				if (abs(dep[dp][ix]-dep[up][ix])>DEPTH_DIFF_LIMIT) break;

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

				// added using depth
				// if ((dep[dp][ix]!=0) && (dep[up][ix]!=0) && (abs(dep[dp][ix]-dep[up][ix])>DEPTH_DIFF_LIMIT)) break;
				if (abs(dep[dp][ix]-dep[up][ix])>DEPTH_DIFF_LIMIT) break;

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

void CFFilter::EnsureSmallestSupportArm(const cv::Mat_<cv::Vec4b> &crMap, int minArm, cv::Mat_<cv::Vec4b> &crMapOut)
{
	crMap.copyTo(crMapOut);
	int height, width, iy, ix;
	height = crMap.rows;
	width = crMap.cols;
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			if (ix >= minArm) crMapOut[iy][ix][0] = std::max<int>(minArm, crMapOut[iy][ix][0]);
			if (ix < width-minArm) crMapOut[iy][ix][2] = std::max<int>(minArm, crMapOut[iy][ix][2]);

			if (iy >= minArm) crMapOut[iy][ix][1] = std::max<int>(minArm, crMapOut[iy][ix][1]);
			if (iy < height-minArm) crMapOut[iy][ix][3] = std::max<int>(minArm, crMapOut[iy][ix][3]);
		}
	}
}

void CFFilter::ExpandCrossToRegionFromSeed( const cv::Mat_<cv::Vec4b> &crMap, cv::Point seed, cv::Mat &labelMask )
{
	int width, height;
	width = crMap.cols;
	height = crMap.rows;

	deque<cv::Point> pQ;
	pQ.clear();

	cv::Mat_<uchar> label = labelMask;

	pQ.push_back(seed);
	label(seed) = 1;
	while (!pQ.empty())
	{
		cv::Point pCur = pQ.front();
		pQ.pop_front();
		int ky, kx;
		cv::Vec4b cross = crMap(pCur);
		for (ky=-cross[1]; ky<=cross[3]; ++ky)
		{
			int ty = ky+pCur.y;
			if (ty<0 || ty>=height) continue;
			if (label[ty][pCur.x] != 1) 
			{
				cv::Point tp(pCur.x, ty);
				label(tp) = 1;
				pQ.push_back(tp);
			}
		}

		for (kx=-cross[0]; kx<=cross[2]; ++kx)
		{
			int tx = kx+pCur.x;
			if (tx<0 || tx>=width) continue;
			if (label[pCur.y][tx] != 1) 
			{
				cv::Point tp(tx, pCur.y);
				label(tp) = 1;
				pQ.push_back(tp);
			}
		}
	}
}

void CFFilter::CountCoverdCross( const cv::Mat_<cv::Vec4b> &crMap, cv::Mat &_ccNum, const char* winName )
{
	int width, height;
	width = crMap.cols;
	height = crMap.rows;

	cv::Mat_<float> ccNum = _ccNum;
	ccNum.create(height, width);
	ccNum.setTo(cv::Scalar(0.0));

	int iy, ix;
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			ccNum[iy][ix] += 1.0;
			int ky, kx;
			cv::Vec4b cross = crMap[iy][ix];
			//for (ky=std::max(iy-cross[1], 0); ky<=std::min(height-1, iy+cross[3]); ++ky)
			for (ky=max(iy-cross[1], 0); ky<=min(height-1, iy+cross[3]); ++ky)
			{
				ccNum[ky][ix] += 1.0;
			}

			//for (kx=std::max(ix-cross[0], 0); kx<=std::min(width-1, ix+cross[2]); ++kx)
			for (kx=max(ix-cross[0], 0); kx<=min(width-1, ix+cross[2]); ++kx)
			{
				ccNum[iy][kx] += 1.0;
			}
		}
	}
	NormalizeFloatAndShow(ccNum, winName);
}

void CFFilter::NormalizeFloatAndShow( const cv::Mat &flMat, const char* winName )
{
	int width, height;
	width = flMat.cols;
	height = flMat.rows;

	int iy, ix;
	float maxVal = 0.0f;

	cv::Mat_<uchar> normFT(height, width);

	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			float tmp = flMat.at<float>(iy, ix);
			if (tmp > maxVal) maxVal = tmp;
		}
	}

	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			float tmp = flMat.at<float>(iy, ix);
			normFT[iy][ix] = (uchar)((tmp/maxVal)*255);
		}
	}

	//cv::Mat binFT;
	//cv::threshold(ftMat, binFT, maxVal/2.0, 255, cv::THRESH_BINARY);
	//char buf[256];
	//sprintf(buf, "%s_THRESH\0", winName);
	//std::string strName;
	//strName.assign(buf);
	//cv::imshow(strName, binFT);

	cv::imshow(winName, normFT);
}

void CFFilter::NormalizeDoubleAndShow(const cv::Mat &doMat, const char* winName)
{
	int width, height;
	width = doMat.cols;
	height = doMat.rows;

	int iy, ix;
	double maxVal = 0.0;

	cv::Mat_<uchar> normDT(height, width);

	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			double tmp = doMat.at<double>(iy, ix);
			if (tmp > maxVal) maxVal = tmp;
		}
	}

	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			double tmp = doMat.at<double>(iy, ix);
			normDT[iy][ix] = (uchar)((tmp/maxVal)*255);
		}
	}

	cv::imshow(winName, normDT);
}

void CFFilter::NormalizeUshortAndShow(const cv::Mat &usMat, const char* winName)
{
	int width, height;
	width = usMat.cols;
	height = usMat.rows;

	int iy, ix;
	ushort maxVal = 0;


	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			ushort tmp = usMat.at<ushort>(iy, ix);
			if (tmp > maxVal) maxVal = tmp;
		}
	}


	cv::Mat_<uchar> normUS(height, width);
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			ushort tmp = usMat.at<ushort>(iy, ix);
			normUS[iy][ix] = (uchar)(tmp*255.0/maxVal);
		}
	}

	//cv::Mat binFT;
	//cv::threshold(ftMat, binFT, maxVal/2.0, 255, cv::THRESH_BINARY);
	//char buf[256];
	//sprintf(buf, "%s_THRESH\0", winName);
	//std::string strName;
	//strName.assign(buf);
	//cv::imshow(strName, binFT);

	cv::imshow(winName, normUS);
}

void CFFilter::CountCoverdCrossRegion(const cv::Mat_<cv::Vec4b> &crMap, cv::Mat &_ccNum, const char* winName)
{
	int width, height;
	width = crMap.cols;
	height = crMap.rows;

	cv::Mat_<float> ccNum = _ccNum;
	ccNum.create(height, width);
	ccNum.setTo(cv::Scalar(0.0));

	int iy, ix;
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			int ky, kx;
			cv::Vec4b pCross = crMap[iy][ix];
			//for (ky=std::max(iy-pCross[1], 0); ky<=std::min(height-1, iy+pCross[3]); ++ky)
			for (ky=max(iy-pCross[1], 0); ky<=min(height-1, iy+pCross[3]); ++ky)
			{
				cv::Vec4b qCross = crMap[ky][ix];
				//for (kx=std::max(ix-qCross[0], 0); kx<=std::min(width-1, ix+qCross[2]); ++kx)
				for (kx=max(ix-qCross[0], 0); kx<=min(width-1, ix+qCross[2]); ++kx)
				{
					ccNum[ky][kx] += 1.0;
				}
			}

			//for (kx=std::max(ix-pCross[0], 0); kx<=std::min(width-1, ix+pCross[2]); ++kx)
			for (kx=max(ix-pCross[0], 0); kx<=min(width-1, ix+pCross[2]); ++kx)
			{
				cv::Vec4b qCross = crMap[iy][kx];
				//for (ky=std::max(iy-qCross[1], 0); ky<=std::min(height-1, iy+qCross[3]); ++ky)
				for (ky=max(iy-qCross[1], 0); ky<=min(height-1, iy+qCross[3]); ++ky)
				{
					ccNum[ky][kx] += 1.0;
				}
			}
		}
	}
	NormalizeFloatAndShow(ccNum, winName);
}

void CFFilter::SummationColorOnCrossMapColMajored(const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat &src, cv::Mat &dst)
{
	////CalcTime ct;
	//ct.Start();

	int iy, ix, width, height;
	width = crMap.cols;
	height = crMap.rows;

	cv::Mat_<cv::Vec3b> color = src;
	cv::Mat_<cv::Vec3d> horSum(height, width+1);

	for (iy=0; iy<height; ++iy)
	{
		cv::Vec3d s(0.0, 0.0, 0.0);
		horSum[iy][0] = s;
		for (ix=0; ix<width; ++ix)
		{
			s += color[iy][ix];
			horSum[iy][ix+1] = s;
		}
	}

	cv::Mat_<cv::Vec3d> crossHorSum(height, width);
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			cv::Vec4b cross = crMap[iy][ix];
			crossHorSum[iy][ix] = horSum[iy][ix+cross[2]+1]-horSum[iy][ix-cross[0]];
		}
	}

	//ct.End("end row process");
	//ct.Start();

	cv::Mat_<cv::Vec3d> verSum(height+1, width);
	for (ix=0; ix<width; ++ix)
	{
		cv::Vec3d s(0.0, 0.0, 0.0);
		verSum[0][ix] = s;
		for (iy=0; iy<height; ++iy)
		{
			s += crossHorSum[iy][ix];
			verSum[iy+1][ix] = s;
		}
	}

	cv::Mat_<cv::Vec3d> crossVerSum(height, width);
	for (ix=0; ix<width; ++ix)
	{
		for (iy=0; iy<height; ++iy)
		{
			cv::Vec4b cross = crMap[iy][ix];
			crossVerSum[iy][ix] = verSum[iy+cross[3]+1][ix]-verSum[iy-cross[1]][ix];
		}
	}

	//ct.End("End Col Process");

	// temporary
	crossVerSum.copyTo(dst);
}

void CFFilter::SummationColorOnCrossMapRowMajored(const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat &src, cv::Mat &dst)
{
	////CalcTime ct;
	//ct.Start();

	int iy, ix, width, height;
	width = crMap.cols;
	height = crMap.rows;

	cv::Mat_<cv::Vec3b> color = src;
	cv::Mat_<cv::Vec3d> verSum(height+1, width);

	for (ix=0; ix<width; ++ix)
	{
		cv::Vec3d s(0.0, 0.0, 0.0);
		verSum[0][ix] = s;
		for (iy=0; iy<height; ++iy)
		{
			s += color[iy][ix];
			verSum[iy+1][ix] = s;
		}
	}

	cv::Mat_<cv::Vec3d> crossVerSum(height, width);
	for (ix=0; ix<width; ++ix)
	{
		for (iy=0; iy<height; ++iy)
		{
			cv::Vec4b cross = crMap[iy][ix];
			crossVerSum[iy][ix] = verSum[iy+cross[3]+1][ix]-verSum[iy-cross[1]][ix];
		}
	}

	//ct.End("end col process");
	//ct.Start();

	cv::Mat_<cv::Vec3d> horSum(height, width+1);
	for (iy=0; iy<height; ++iy)
	{
		cv::Vec3d s(0.0, 0.0, 0.0);
		horSum[iy][0] = s;
		for (ix=0; ix<width; ++ix)
		{
			s += crossVerSum[iy][ix];
			horSum[iy][ix+1] = s;
		}
	}

	cv::Mat_<cv::Vec3d> crossHorSum(height, width);
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)		
		{
			cv::Vec4b cross = crMap[iy][ix];
			crossHorSum[iy][ix] = horSum[iy][ix+cross[2]+1]-horSum[iy][ix-cross[0]];
		}
	}

	//ct.End("End Row Process");

	// temporary
	crossHorSum.copyTo(dst);
}

void CFFilter::SummationColorOnRect(const cv::Mat &src, cv::Mat &_dst, int rectSize)
{
	int width, height, iy, ix;
	width = src.cols;
	height = src.rows;

	cv::Mat_<cv::Vec3b> color = src;
	cv::Mat_<cv::Vec3d> colorSum(height+1, width+1);

	cv::Vec3d s(0.0, 0.0, 0.0);
	for (ix=0; ix<width+1; ++ix)
	{
		colorSum[0][ix] = s;
	}
	for (iy=0; iy<height+1; ++iy)
	{
		colorSum[iy][0] = s;
	}
	for (iy=1; iy<=height; ++iy)
	{
		//s.all(0.0); 
		//s.zeros();
		s[0] = s[1] = s[2] = 0.0;
		for (ix=1; ix<=width; ++ix)
		{
			s += color[iy-1][ix-1];
			colorSum[iy][ix] = colorSum[iy-1][ix]+s;
		}
	}

	_dst.create(height, width, CV_64FC3);
	cv::Mat_<cv::Vec3d> dst = _dst;

	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			int left, right, up, down;
			/*right = std::min(ix+1+rectSize, width);
			left = std::max(ix-rectSize, 0);
			up = std::max(iy-rectSize, 0);
			down = std::min(iy+1+rectSize, height);*/
			right = min(ix+1+rectSize, width);
			left = max(ix-rectSize, 0);
			up = max(iy-rectSize, 0);
			down = min(iy+1+rectSize, height);
			dst[iy][ix] = colorSum[down][right]+colorSum[up][left]-colorSum[down][left]-colorSum[up][right];
		}
	}
}

void CFFilter::Summation3CDoubleOnCrossMapColMajored(const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat &src, cv::Mat &dst)
{
	////CalcTime ct;
	//ct.Start();

	int iy, ix, width, height;
	width = crMap.cols;
	height = crMap.rows;

	cv::Mat_<cv::Vec3d> color = src;
	cv::Mat_<cv::Vec3d> horSum(height, width+1);

	for (iy=0; iy<height; ++iy)
	{
		cv::Vec3d s(0.0, 0.0, 0.0);
		horSum[iy][0] = s;
		for (ix=0; ix<width; ++ix)
		{
			s += color[iy][ix];
			horSum[iy][ix+1] = s;
		}
	}

	cv::Mat_<cv::Vec3d> crossHorSum(height, width);
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			cv::Vec4b cross = crMap[iy][ix];
			crossHorSum[iy][ix] = horSum[iy][ix+cross[2]+1]-horSum[iy][ix-cross[0]];
		}
	}

	//ct.End("end row process");
	//ct.Start();

	cv::Mat_<cv::Vec3d> verSum(height+1, width);
	for (ix=0; ix<width; ++ix)
	{
		cv::Vec3d s(0.0, 0.0, 0.0);
		verSum[0][ix] = s;
		for (iy=0; iy<height; ++iy)
		{
			s += crossHorSum[iy][ix];
			verSum[iy+1][ix] = s;
		}
	}

	cv::Mat_<cv::Vec3d> crossVerSum(height, width);
	for (ix=0; ix<width; ++ix)
	{
		for (iy=0; iy<height; ++iy)
		{
			cv::Vec4b cross = crMap[iy][ix];
			crossVerSum[iy][ix] = verSum[iy+cross[3]+1][ix]-verSum[iy-cross[1]][ix];
		}
	}

	//ct.End("End Col Process");

	// temporary
	crossVerSum.copyTo(dst);
}

void CFFilter::CountSizeOnCrossMapColMajored(const cv::Mat_<cv::Vec4b> &crMap, cv::Mat &count)
{
	int iy, ix, width, height;
	width = crMap.cols;
	height = crMap.rows;

	cv::Mat_<int> crossHorSum(height, width);
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			cv::Vec4b cross = crMap[iy][ix];
			crossHorSum[iy][ix] = cross[2]+cross[0]+1;
		}
	}

	cv::Mat_<int> verSum(height+1, width);
	for (ix=0; ix<width; ++ix)
	{
		int s = 0;
		verSum[0][ix] = 0;
		for (iy=0; iy<height; ++iy)
		{
			s += crossHorSum[iy][ix];
			verSum[iy+1][ix] = s;
		}
	}

	count.create(height, width, CV_16UC1);
	cv::Mat_<ushort> crossVerSum = count;
	for (ix=0; ix<width; ++ix)
	{
		for (iy=0; iy<height; ++iy)
		{
			cv::Vec4b cross = crMap[iy][ix];
			crossVerSum[iy][ix] = (ushort)(verSum[iy+cross[3]+1][ix]-verSum[iy-cross[1]][ix]);
		}
	}
}

void CFFilter::CountSizeOnCrossMapRowMajored(const cv::Mat_<cv::Vec4b> &crMap, cv::Mat &count)
{
	int iy, ix, width, height;
	width = crMap.cols;
	height = crMap.rows;

	cv::Mat_<int> crossVerSum(height, width);
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			cv::Vec4b cross = crMap[iy][ix];
			crossVerSum[iy][ix] = cross[3]+cross[1]+1;
		}
	}

	cv::Mat_<int> horSum(height, width+1);
	for (iy=0; iy<height; ++iy)
	{
		int s = 0;
		horSum[iy][0] = 0;
		for (ix=0; ix<width; ++ix)
		{
			s += crossVerSum[iy][ix];
			horSum[iy][ix+1] = s;
		}
	}

	count.create(height, width, CV_16UC1);
	cv::Mat_<ushort> crossHorSum = count;
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			cv::Vec4b cross = crMap[iy][ix];
			crossHorSum[iy][ix] = (ushort)(horSum[iy][ix+cross[2]+1]-horSum[iy][ix-cross[0]]);
		}
	}
}

void CFFilter::CountSizeOnRect(const cv::Mat_<cv::Vec4b> &crMap, cv::Mat &count, int rectSize)
{
	int width, height, iy, ix;
	width = crMap.cols;
	height = crMap.rows;

	cv::Mat_<int> sizeSum(height+1, width+1);

	int s = 0;
	for (ix=0; ix<width+1; ++ix)
	{
		sizeSum[0][ix] = 0;
	}
	for (iy=1; iy<=height; ++iy)
	{
		for (ix=0; ix<=width; ++ix)
		{
			sizeSum[iy][ix] = sizeSum[iy-1][ix]+ix;
		}
	}

	count.create(height, width, CV_16UC1);
	cv::Mat_<ushort> dst = count;

	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			int left, right, up, down;
			/*right = std::min(ix+1+rectSize, width);
			left = std::max(ix-rectSize, 0);
			up = std::max(iy-rectSize, 0);
			down = std::min(iy+1+rectSize, height);*/
			right = min(ix+1+rectSize, width);
			left = max(ix-rectSize, 0);
			up = max(iy-rectSize, 0);
			down = min(iy+1+rectSize, height);
			dst[iy][ix] = (ushort)(sizeSum[down][right]+sizeSum[up][left]-sizeSum[down][left]-sizeSum[up][right]);
		}
	}
}


void CFFilter::AverageImageUsingCrossMap(const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat &src, cv::Mat &dst)
{
	int height, width, iy, ix;
	height = crMap.rows;
	width = crMap.cols;
	cv::Mat _crossVerSum;
	SummationColorOnCrossMapColMajored(crMap, src, _crossVerSum);
	cv::Mat_<cv::Vec3d> crossVerSum = _crossVerSum;
	////CalcTime ct;
	//ct.Start();

	cv::Mat crossSize;
	CountSizeOnCrossMapColMajored(crMap, crossSize);

	dst.create(height, width, CV_8UC3);
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			dst.at<cv::Vec3b>(iy, ix)[0] = (uchar)(crossVerSum[iy][ix][0]/(double)crossSize.at<ushort>(iy, ix));
			dst.at<cv::Vec3b>(iy, ix)[1] = (uchar)(crossVerSum[iy][ix][1]/(double)crossSize.at<ushort>(iy, ix));
			dst.at<cv::Vec3b>(iy, ix)[2] = (uchar)(crossVerSum[iy][ix][2]/(double)crossSize.at<ushort>(iy, ix));
		}
	}
	//ct.End("end of average color");
}

void CFFilter::CalacuteCrossRectDifference(const cv::Mat &img, int maxL, int tau, cv::Mat &diffRes)
{
	int height, width, iy, ix;
	height = img.rows;
	width = img.cols;

	cv::Mat_<cv::Vec4b> crMap;
	GetCrossUsingSlidingWindow(img, crMap, maxL, tau);
	cv::Mat crossSum;
	SummationColorOnCrossMapRowMajored(crMap, img, crossSum);

	cv::Mat rectSum;
	SummationColorOnRect(img, rectSum, maxL);

	cv::Mat crossSize;
	CountSizeOnCrossMapRowMajored(crMap, crossSize);

	diffRes.create(height, width, CV_32FC1);
	cv::Mat_<float> res = diffRes;
	const double TEMP_MAX = 800.0;
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			int pRegion = crossSize.at<ushort>(iy, ix);
			if (pRegion < 2*maxL)
			{
				res[iy][ix] = (float)TEMP_MAX;
			}
			else
			{
				cv::Vec3d inner = crossSum.at<cv::Vec3d>(iy, ix);
				cv::Vec3d whole = rectSum.at<cv::Vec3d>(iy, ix);
				double wholeSize = (2*maxL+1)*(2*maxL+1);
				double cr = fabs(whole[0]-inner[0])+fabs(whole[1]-inner[1])+fabs(whole[2]-inner[2]);
				res[iy][ix] = (float)(min(TEMP_MAX, cr/(wholeSize-pRegion)));
				//res[iy][ix] = (float)(std::min(TEMP_MAX, cr/(wholeSize-pRegion)));
			}
		}
	}
	NormalizeFloatAndShow(res, "FloatShow");
}

void CFFilter::DrawCrossArmPlot(const cv::Mat_<cv::Vec4b> &crMap, int row, int col)
{
	int width, height, iy, ix;
	height = crMap.rows;
	width = crMap.cols;

	cv::Mat_<cv::Vec3b> canvas;
	const int CANVAS_HEIGHT = 500;
	const int COL_COEF = 1;
	canvas.create(CANVAS_HEIGHT+1, COL_COEF*width);
	canvas.setTo(255.0);
	for (ix=0; ix<width; ++ix)
		cv::line(canvas, cv::Point(ix*COL_COEF, CANVAS_HEIGHT), cv::Point(ix*COL_COEF, CANVAS_HEIGHT-(crMap[row][ix][2]+crMap[row][ix][0])*10), cv::Scalar(255, 0, 0));
	cv::imshow("Row_Cross_Plot", canvas);
	
	const int ROW_COEF = 1;
	const int CANVAS_WIDTH = 500;
	canvas.create(ROW_COEF*height, CANVAS_WIDTH+1);
	canvas.setTo(255.0);
	for (iy=0; iy<height; ++iy)
		cv::line(canvas, cv::Point(0, iy*ROW_COEF), cv::Point((crMap[iy][col][1]+crMap[iy][col][3])*10, iy*ROW_COEF), cv::Scalar(255, 0, 0));
	cv::imshow("Col_Cross_Plot", canvas);
}


void CFFilter::SummationDoubleOnCrossMapColMajored(const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat &src, cv::Mat &dst)
{
	////CalcTime ct;
	//ct.Start();

	int iy, ix, width, height;
	width = crMap.cols;
	height = crMap.rows;

	cv::Mat_<double> cost = src;
	cv::Mat_<double> horSum(height, width+1);

	for (iy=0; iy<height; ++iy)
	{
		double s = 0.0;
		horSum[iy][0] = s;
		for (ix=0; ix<width; ++ix)
		{
			s += cost[iy][ix];
			horSum[iy][ix+1] = s;
		}
	}

	cv::Mat_<double> crossHorSum(height, width);
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			cv::Vec4b cross = crMap[iy][ix];
			crossHorSum[iy][ix] = horSum[iy][ix+cross[2]+1]-horSum[iy][ix-cross[0]];
		}
	}

	cv::Mat_<double> verSum(height+1, width);
	for (ix=0; ix<width; ++ix)
	{
		double s = 0.0;
		verSum[0][ix] = s;
		for (iy=0; iy<height; ++iy)
		{
			s += crossHorSum[iy][ix];
			verSum[iy+1][ix] = s;
		}
	}

	cv::Mat_<double> crossVerSum(height, width);
	for (ix=0; ix<width; ++ix)
	{
		for (iy=0; iy<height; ++iy)
		{
			cv::Vec4b cross = crMap[iy][ix];
			//printf("%d %d %d %d %d %d\n", height, width, cross[0], cross[1], cross[2], cross[3]);
			crossVerSum[iy][ix] = verSum[iy+cross[3]+1][ix]-verSum[iy-cross[1]][ix];
		}
	}

	// temporary
	crossVerSum.copyTo(dst);

	//ct.End("End Summation of double by CrossMap(Col Majored) Process");
}

void CFFilter::SummationDoubleOnCrossMapRowMajored(const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat &src, cv::Mat &dst)
{
	////CalcTime ct;
	//ct.Start();

	int iy, ix, width, height;
	width = crMap.cols;
	height = crMap.rows;

	cv::Mat_<double> cost = src;
	cv::Mat_<double> verSum(height+1, width);

	for (ix=0; ix<width; ++ix)
	{
		double s = 0.0;
		verSum[0][ix] = s;
		for (iy=0; iy<height; ++iy)
		{
			s += cost[iy][ix];
			verSum[iy+1][ix] = s;
		}
	}

	cv::Mat_<double> crossVerSum(height, width);
	for (ix=0; ix<width; ++ix)
	{
		for (iy=0; iy<height; ++iy)
		{
			cv::Vec4b cross = crMap[iy][ix];
			crossVerSum[iy][ix] = verSum[iy+cross[3]+1][ix]-verSum[iy-cross[1]][ix];
		}
	}

	cv::Mat_<double> horSum(height, width+1);
	for (iy=0; iy<height; ++iy)
	{
		double s = 0.0;
		horSum[iy][0] = s;
		for (ix=0; ix<width; ++ix)
		{
			s += crossVerSum[iy][ix];
			horSum[iy][ix+1] = s;
		}
	}

	cv::Mat_<double> crossHorSum(height, width);
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			cv::Vec4b cross = crMap[iy][ix];
			crossHorSum[iy][ix] = horSum[iy][ix+cross[2]+1]-horSum[iy][ix-cross[0]];
		}
	}

	// temporary
	crossHorSum.copyTo(dst);

	//ct.End("End Summation of double by CrossMap(Row Majored) Process");
}


void CFFilter::SummationDoubleOnRect(const cv::Mat &src, cv::Mat &_dst, int rectSize)
{
	////CalcTime ct;
	//ct.Start();

	int width, height, iy, ix;
	width = src.cols;
	height = src.rows;

	cv::Mat_<double> cost = src;
	cv::Mat_<double> costSum(height+1, width+1);

	double s = 0.0;
	for (ix=0; ix<width+1; ++ix)
	{
		costSum[0][ix] = s;
	}
	for (iy=0; iy<height+1; ++iy)
	{
		costSum[iy][0] = s;
	}
	for (iy=1; iy<=height; ++iy)
	{
		s = 0.0;
		for (ix=1; ix<=width; ++ix)
		{
			s += cost[iy-1][ix-1];
			costSum[iy][ix] = costSum[iy-1][ix]+s;
		}
	}

	_dst.create(height, width, CV_64FC1);
	cv::Mat_<double> dst = _dst;

	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			int left, right, up, down;
			/*right = std::min(ix+1+rectSize, width);
			left = std::max(ix-rectSize, 0);
			up = std::max(iy-rectSize, 0);
			down = std::min(iy+1+rectSize, height);*/
			right = min(ix+1+rectSize, width);
			left = max(ix-rectSize, 0);
			up = max(iy-rectSize, 0);
			down = min(iy+1+rectSize, height);
			dst[iy][ix] = costSum[down][right]+costSum[up][left]-costSum[down][left]-costSum[up][right];
		}
	}
	//ct.End("End Summation of double[Rect] Process");
}


void CFFilter::FilterColorImageUsingCrossMapColMajored(const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat &src, cv::Mat &dst, int maxL, int tau, double lambda)
{
	////CalcTime ct;
	//ct.Start();
	int height, width, iy, ix;
	height = src.rows;
	width = src.cols;


	cv::Mat crossSum;
	SummationColorOnCrossMapColMajored(crMap, src, crossSum);

	cv::Mat rectSum;
	SummationColorOnRect(src, rectSum, maxL);

	cv::Mat crossSize;
	CountSizeOnCrossMapColMajored(crMap, crossSize);

	cv::Mat rectSize;
	CountSizeOnRect(crMap, rectSize, maxL);

	dst.create(height, width, CV_8UC3);
	cv::Mat_<cv::Vec3b> res = dst;
	const double LAMBDA = lambda;
	//const double RECT_SIZE = (2*maxL+1)*(2*maxL+1);
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			int pRegion = crossSize.at<ushort>(iy, ix);
			int rectRegion = rectSize.at<ushort>(iy, ix);
			const double NORMALIZE_FACTOR = (1.0-LAMBDA)*pRegion+LAMBDA*rectRegion;

			cv::Vec3d inner = crossSum.at<cv::Vec3d>(iy, ix);
			cv::Vec3d whole = rectSum.at<cv::Vec3d>(iy, ix);

			//res[iy][ix] = inner+LAMBDA*(whole-inner);
			cv::Vec3d w = (1.0-LAMBDA)*inner+LAMBDA*whole;
			res[iy][ix][0] = (uchar)(w[0]/NORMALIZE_FACTOR);
			res[iy][ix][1] = (uchar)(w[1]/NORMALIZE_FACTOR);
			res[iy][ix][2] = (uchar)(w[2]/NORMALIZE_FACTOR);

		}
	}
	//ct.End("Color Image(Col Majored) filter once");
}

void CFFilter::FilterColorImageUsingCrossMapRowMajored(const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat &src, cv::Mat &dst, int maxL, int tau, double lambda)
{
	////CalcTime ct;
	//ct.Start();
	int height, width, iy, ix;
	height = src.rows;
	width = src.cols;


	cv::Mat crossSum;
	SummationColorOnCrossMapRowMajored(crMap, src, crossSum);

	cv::Mat rectSum;
	SummationColorOnRect(src, rectSum, maxL);

	cv::Mat crossSize;
	CountSizeOnCrossMapRowMajored(crMap, crossSize);

	cv::Mat rectSize;
	CountSizeOnRect(crMap, rectSize, maxL);

	dst.create(height, width, CV_8UC3);
	cv::Mat_<cv::Vec3b> res = dst;
	const double LAMBDA = lambda;
	//const double RECT_SIZE = (2*maxL+1)*(2*maxL+1);
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			int pRegion = crossSize.at<ushort>(iy, ix);
			int rectRegion = rectSize.at<ushort>(iy, ix);
			const double NORMALIZE_FACTOR = (1.0-LAMBDA)*pRegion+LAMBDA*rectRegion;

			cv::Vec3d inner = crossSum.at<cv::Vec3d>(iy, ix);
			cv::Vec3d whole = rectSum.at<cv::Vec3d>(iy, ix);

			//res[iy][ix] = inner+LAMBDA*(whole-inner);
			cv::Vec3d w = (1.0-LAMBDA)*inner+LAMBDA*whole;
			//res[iy][ix] = w/NORMALIZE_FACTOR;
			res[iy][ix][0] = (uchar)(w[0]/NORMALIZE_FACTOR);
			res[iy][ix][1] = (uchar)(w[1]/NORMALIZE_FACTOR);
			res[iy][ix][2] = (uchar)(w[2]/NORMALIZE_FACTOR);

		}
	}
	//ct.End("Color Image(Row Majored) filter once");
}

void CFFilter::FilterColorImageUsingCrossMapRowAndCol(const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat &src, cv::Mat &dst, int maxL, int tau, double lambda)
{
	////CalcTime //ct;
	//ct.Start();
	int height, width, iy, ix;
	height = src.rows;
	width = src.cols;


	cv::Mat crossSumCol, crossSumRow;
	SummationColorOnCrossMapRowMajored(crMap, src, crossSumRow);
	SummationColorOnCrossMapColMajored(crMap, src, crossSumCol);

	cv::Mat rectSum;
	SummationColorOnRect(src, rectSum, maxL);

	cv::Mat crossSizeRow, crossSizeCol;
	CountSizeOnCrossMapRowMajored(crMap, crossSizeRow);
	CountSizeOnCrossMapColMajored(crMap, crossSizeCol);

	cv::Mat rectSize;
	CountSizeOnRect(crMap, rectSize, maxL);

	dst.create(height, width, CV_8UC3);
	cv::Mat_<cv::Vec3b> res = dst;
	const double LAMBDA = lambda;
	//const double RECT_SIZE = (2*maxL+1)*(2*maxL+1);
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			int pRegionRow = crossSizeRow.at<ushort>(iy, ix);
			int pRegionCol = crossSizeCol.at<ushort>(iy, ix);
			int rectRegion = rectSize.at<ushort>(iy, ix);
			const double NORMALIZE_FACTOR = (1.0-LAMBDA)*(pRegionRow+pRegionCol)+LAMBDA*2*rectRegion;

			cv::Vec3d innerRow = crossSumRow.at<cv::Vec3d>(iy, ix);
			cv::Vec3d innerCol = crossSumCol.at<cv::Vec3d>(iy, ix);
			cv::Vec3d whole = rectSum.at<cv::Vec3d>(iy, ix);

			//res[iy][ix] = inner+LAMBDA*(whole-inner);
			cv::Vec3d w = (1.0-LAMBDA)*(innerRow+innerCol)+LAMBDA*2*whole;
			//res[iy][ix] = w/NORMALIZE_FACTOR;
			res[iy][ix][0] = (uchar)(w[0]/NORMALIZE_FACTOR);
			res[iy][ix][1] = (uchar)(w[1]/NORMALIZE_FACTOR);
			res[iy][ix][2] = (uchar)(w[2]/NORMALIZE_FACTOR);
		}
	}
	//ct.End("Color Image(Row and Col) filter once");
}


void CFFilter::FilterCostVolumeByCrossMapColMajored(const cv::Mat &costVolume, const cv::Mat_<cv::Vec4b> &crMap, cv::Mat &filteredVolume, int maxL, int tau, double lambda)
{
	////CalcTime ct;
	//ct.Start();
	int height, width, iy, ix;
	height = costVolume.rows;
	width = costVolume.cols;

	cv::Mat crossSum;
	SummationDoubleOnCrossMapColMajored(crMap, costVolume, crossSum);

	cv::Mat rectSum;
	SummationDoubleOnRect(costVolume, rectSum, maxL);

	cv::Mat crossSize;
	CountSizeOnCrossMapColMajored(crMap, crossSize);

	cv::Mat rectSize;
	CountSizeOnRect(crMap, rectSize, maxL);

	filteredVolume.create(height, width, CV_64FC1);
	cv::Mat_<double> res = filteredVolume;
	const double LAMBDA = lambda/100.0;
	//const double RECT_SIZE = (2*maxL+1)*(2*maxL+1);
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			int pRegion = crossSize.at<ushort>(iy, ix);
			int rectRegion = rectSize.at<ushort>(iy, ix);
			const double NORMALIZE_FACTOR = (1.0-LAMBDA)*pRegion+LAMBDA*rectRegion;

			double inner = crossSum.at<double>(iy, ix);
			double whole = rectSum.at<double>(iy, ix);

			//res[iy][ix] = inner+LAMBDA*(whole-inner);
			double w = (1.0-LAMBDA)*inner+LAMBDA*whole;
			res[iy][ix] = w/NORMALIZE_FACTOR;
		}
	}
	//ct.End("Cost volume(double) (Col-Majored) filter once");
}

void CFFilter::FilterCostVolumeByCrossMapRowMajored(const cv::Mat &costVolume, const cv::Mat_<cv::Vec4b> &crMap, cv::Mat &filteredVolume, int maxL, int tau, double lambda)
{
	////CalcTime ct;
	//ct.Start();
	int height, width, iy, ix;
	height = costVolume.rows;
	width = costVolume.cols;

	cv::Mat crossSum;
	SummationDoubleOnCrossMapRowMajored(crMap, costVolume, crossSum);

	cv::Mat rectSum;
	SummationDoubleOnRect(costVolume, rectSum, maxL);

	cv::Mat crossSize;
	CountSizeOnCrossMapRowMajored(crMap, crossSize);

	cv::Mat rectSize;
	CountSizeOnRect(crMap, rectSize, maxL);

	filteredVolume.create(height, width, CV_64FC1);
	cv::Mat_<double> res = filteredVolume;
	const double LAMBDA = lambda;
	//const double RECT_SIZE = (2*maxL+1)*(2*maxL+1);
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			int pRegion = crossSize.at<ushort>(iy, ix);
			int rectRegion = rectSize.at<ushort>(iy, ix);
			const double NORMALIZE_FACTOR = (1.0-LAMBDA)*pRegion+LAMBDA*rectRegion;

			double inner = crossSum.at<double>(iy, ix);
			double whole = rectSum.at<double>(iy, ix);

			//res[iy][ix] = inner+LAMBDA*(whole-inner);
			double w = (1.0-LAMBDA)*inner+LAMBDA*whole;
			res[iy][ix] = w/NORMALIZE_FACTOR;
		}
	}
	//ct.End("Cost volume(double) (Row-Majored) filter once");
}

void CFFilter::FilterCostVolumeByCrossMapRowAndCol(const cv::Mat &costVolume, const cv::Mat_<cv::Vec4b> &crMap, cv::Mat &filteredVolume, int maxL, int tau, double lambda)
{
	////CalcTime ct;
	//ct.Start();
	int height, width, iy, ix;
	height = costVolume.rows;
	width = costVolume.cols;

	cv::Mat crossSumCol, crossSumRow;
	SummationDoubleOnCrossMapColMajored(crMap, costVolume, crossSumCol);
	SummationDoubleOnCrossMapRowMajored(crMap, costVolume, crossSumRow);

	cv::Mat rectSum;
	SummationDoubleOnRect(costVolume, rectSum, maxL);

	cv::Mat crossSizeCol, crossSizeRow;
	CountSizeOnCrossMapColMajored(crMap, crossSizeCol);
	CountSizeOnCrossMapRowMajored(crMap, crossSizeRow);

	cv::Mat rectSize;
	CountSizeOnRect(crMap, rectSize, maxL);

	filteredVolume.create(height, width, CV_64FC1);
	cv::Mat_<double> res = filteredVolume;
	const double LAMBDA = lambda;
	//const double RECT_SIZE = (2*maxL+1)*(2*maxL+1);
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			int pRegionRow = crossSizeRow.at<ushort>(iy, ix);
			int pRegionCol = crossSizeCol.at<ushort>(iy, ix);
			int rectRegion = rectSize.at<ushort>(iy, ix);
			const double NORMALIZE_FACTOR = (1.0-LAMBDA)*(pRegionRow+pRegionCol)+LAMBDA*rectRegion;

			double innerCol = crossSumCol.at<double>(iy, ix);
			double innerRow = crossSumRow.at<double>(iy, ix);
			double whole = rectSum.at<double>(iy, ix);

			//res[iy][ix] = inner+LAMBDA*(whole-inner);
			double w = (1.0-LAMBDA)*(innerCol+innerRow)+LAMBDA*whole;
			res[iy][ix] = w/NORMALIZE_FACTOR;
		}
	}
	//ct.End("Cost volume(double) (Row and Col) filter once");
}

void CFFilter::GetCrossUsingSlidingWindowInLUV(const cv::Mat_<cv::Vec3f> &img, cv::Mat_<cv::Vec4b> &crMap, int maxL, int tau)
{
	if ((img.data == NULL) || img.empty()) return;

	////CalcTime ct;
	//ct.Start();

	int width, height;
	width = img.cols;
	height = img.rows;

	crMap.create(height, width);
	cv::Point p;
	for (p.y=0; p.y<height; ++p.y)
	{
		// to determine right most, cr[2]
		cv::Vec3f sumColor;
		int lp, rp;
		rp = 0;
		for (lp=0; lp<width; ++lp)
		{
			int diff = rp-lp;
			if (diff == 0)
			{
				sumColor = img[p.y][rp];
				++rp;
				++diff;
			}

			while (diff <= maxL)
			{
				if (rp >= width) break;

				if (fabs(sumColor[0]/diff-img[p.y][rp][0])>tau || fabs(sumColor[1]/diff-img[p.y][rp][1])>tau
					|| fabs(sumColor[2]/diff-img[p.y][rp][2])>tau) break;
				sumColor += img[p.y][rp];
				++diff;
				++rp;
			}
			sumColor -= img[p.y][lp];

			crMap[p.y][lp][2] = diff-1;

			//printf("row = %d\t%d %d %d\n", p.y, lp, rp, diff-1);
		}

		// to determine left most, cr[0]
		lp = width-1;
		for (rp=width-1; rp>=0; --rp)
		{
			int diff = rp-lp;
			if (diff == 0)
			{
				sumColor = img[p.y][lp];
				--lp;
				++diff;
			}
			while (diff <= maxL)
			{
				if (lp < 0) break;
				if (fabs(sumColor[0]/diff-img[p.y][lp][0])>tau || fabs(sumColor[1]/diff-img[p.y][lp][1])>tau
					|| fabs(sumColor[2]/diff-img[p.y][lp][2])>tau) break;
				sumColor += img[p.y][lp];
				++diff;
				--lp;
			}

			sumColor -= img[p.y][rp];

			crMap[p.y][rp][0] = diff-1;
		}

	}

	// determine up and down arm length
	for (p.x=0; p.x<width; ++p.x)
	{
		// to determine down most, cr[3]
		cv::Vec3f sumColor;
		int up, dp;
		dp = 0;
		for (up=0; up<height; ++up)
		{
			int diff = dp-up;
			if (diff == 0)
			{
				sumColor = img[dp][p.x];
				++dp;
				++diff;
			}
			while (diff <= maxL)
			{
				if (dp >= height) break;
				if (fabs(sumColor[0]/diff-img[dp][p.x][0])>tau || fabs(sumColor[1]/diff-img[dp][p.x][1])>tau
					|| fabs(sumColor[2]/diff-img[dp][p.x][2])>tau) break;
				sumColor += img[dp][p.x];
				++diff;
				++dp;
			}

			sumColor -= img[up][p.x];

			crMap[up][p.x][3] = diff-1;
		}

		// to determine up most, cr[1]
		up = height-1;
		for (dp=height-1; dp>=0; --dp)
		{
			int diff = dp-up;
			if (diff == 0)
			{
				sumColor = img[up][p.x];
				--up;
				++diff;
			}

			while (diff <= maxL)
			{
				if (up < 0) break;
				if (fabs(sumColor[0]/diff-img[up][p.x][0])>tau || fabs(sumColor[1]/diff-img[up][p.x][1])>tau
					|| fabs(sumColor[2]/diff-img[up][p.x][2])>tau) break;
				sumColor += img[up][p.x];
				++diff;
				--up;
			}

			sumColor -= img[dp][p.x];

			crMap[dp][p.x][1] = diff-1;
		}
	}

	//ct.End("SlWin(LUV) Construct CrossMap");
}


void CFFilter::CLMFilter0ColorImageColMajored(const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat &src, cv::Mat &dst, int maxL, int tau)
{
	////CalcTime ct;
	//ct.Start();
	int height, width, iy, ix;
	height = src.rows;
	width = src.cols;


	cv::Mat crossSum;
	SummationColorOnCrossMapColMajored(crMap, src, crossSum);

	cv::Mat crossSize;
	CountSizeOnCrossMapColMajored(crMap, crossSize);

	cv::Mat_<ushort> crSize = crossSize;
	cv::Mat_<double> interSize(height, width);
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			interSize[iy][ix] = (double)(crSize[iy][ix]);
		}
	}

	Summation3CDoubleOnCrossMapColMajored(crMap, crossSum, crossSum);

	SummationDoubleOnCrossMapColMajored(crMap, interSize, interSize);

	dst.create(height, width, CV_8UC3);
	cv::Mat_<cv::Vec3b> res = dst;
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			double pRegion = interSize.at<double>(iy, ix);
			cv::Vec3d inner = crossSum.at<cv::Vec3d>(iy, ix);

			res[iy][ix][0] = (uchar)(inner[0]/pRegion);
			res[iy][ix][1] = (uchar)(inner[1]/pRegion);
			res[iy][ix][2] = (uchar)(inner[2]/pRegion);
		}
	}
	//ct.End("CLMF0(Col Majored) filter once");
}

void CFFilter::CLMFilter0CostVolumeColMajored(const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat &src, cv::Mat &dst, int maxL, int tau)
{
	////CalcTime ct;
	//ct.Start();
	int height, width, iy, ix;
	height = src.rows;
	width = src.cols;


	cv::Mat crossSum;
	SummationDoubleOnCrossMapColMajored(crMap, src, crossSum);

	cv::Mat crossSize;
	CountSizeOnCrossMapColMajored(crMap, crossSize);

	cv::Mat_<ushort> crSize = crossSize;
	cv::Mat_<double> interSize(height, width);
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			interSize[iy][ix] = (double)(crSize[iy][ix]);
		}
	}

	SummationDoubleOnCrossMapColMajored(crMap, crossSum, crossSum);

	SummationDoubleOnCrossMapColMajored(crMap, interSize, interSize);

	dst.create(height, width, CV_64FC1);
	cv::Mat_<double> res = dst;
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			double pRegion = interSize.at<double>(iy, ix);
			double inner = crossSum.at<double>(iy, ix);

			res[iy][ix] = inner/pRegion;
		}
	}
	//ct.End("CLMF0 Cost Volume(Col Majored) filter once");
}

void CFFilter::WeightedFilterColorImageColMajored(const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat &src, cv::Mat &dst, int maxL, int tau)
{
	////CalcTime ct;
	//ct.Start();
	int height, width, iy, ix;
	height = src.rows;
	width = src.cols;

	
//	SummationColorOnCrossMapColMajored(crMap, src, crossSum);

	cv::Mat crossSize;
	CountSizeOnCrossMapColMajored(crMap, crossSize);

	cv::Mat_<ushort> crSize = crossSize;
	cv::Mat_<cv::Vec3d> weightImg(height, width);
	cv::Mat_<cv::Vec3b> img = src;
	cv::Mat_<double> interSize(height, width);
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			interSize[iy][ix] = (double)(crSize[iy][ix]);
			weightImg[iy][ix][0] = img[iy][ix][0]*interSize[iy][ix];
			weightImg[iy][ix][1] = img[iy][ix][1]*interSize[iy][ix];
			weightImg[iy][ix][2] = img[iy][ix][2]*interSize[iy][ix];
		}
	}

	cv::Mat crossSum;
	Summation3CDoubleOnCrossMapColMajored(crMap, weightImg, crossSum);

	SummationDoubleOnCrossMapColMajored(crMap, interSize, interSize);

	dst.create(height, width, CV_8UC3);
	cv::Mat_<cv::Vec3b> res = dst;
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			double pRegion = interSize.at<double>(iy, ix);
			cv::Vec3d inner = crossSum.at<cv::Vec3d>(iy, ix);

			res[iy][ix][0] = (uchar)(inner[0]/pRegion);
			res[iy][ix][1] = (uchar)(inner[1]/pRegion);
			res[iy][ix][2] = (uchar)(inner[2]/pRegion);
		}
	}
	//ct.End("Weighted on Original Image(Col Majored) filter once");
}

void CFFilter::CalculateGradientBasedOnCrossmap(const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat &_img)
{
	int iy, ix, height, width;
	height = crMap.rows;
	width = crMap.cols;
	
	cv::Mat_<cv::Vec3b> img = _img;
	cv::Mat_<cv::Vec3d> horSum(height, width+1);

	for (iy=0; iy<height; ++iy)
	{
		cv::Vec3d s(0.0, 0.0, 0.0);
		horSum[iy][0] = s;
		for (ix=0; ix<width; ++ix)
		{
			s += img[iy][ix];
			horSum[iy][ix+1] = s;
		}
	}

	cv::Mat_<cv::Vec3d> aveHorSum(height, width);
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			cv::Vec4b cross = crMap[iy][ix];
			double size = (double)cross[2]+1.0+cross[0];
			aveHorSum[iy][ix][0] = (horSum[iy][ix+cross[2]+1][0]-horSum[iy][ix-cross[0]][0])/(size);
			aveHorSum[iy][ix][1] = (horSum[iy][ix+cross[2]+1][1]-horSum[iy][ix-cross[0]][1])/(size);
			aveHorSum[iy][ix][2] = (horSum[iy][ix+cross[2]+1][2]-horSum[iy][ix-cross[0]][2])/(size);
			//printf("%f\n", aveHorSum[iy][ix][0]);
		}
	}

	cv::Mat_<cv::Vec3d> gradX(height, width);
	for (iy=0; iy<height; ++iy)
	{
		gradX[iy][0][0] = 0.0;
		gradX[iy][0][1] = 0.0;
		gradX[iy][0][2] = 0.0;
		for (ix=1; ix<width; ++ix)
		{
			gradX[iy][ix] = aveHorSum[iy][ix]-aveHorSum[iy][ix-1];
			//printf("%f\n", gradX[iy][ix][0]);
		}
	}

	cv::Mat_<cv::Vec3d> verSum(height+1, width);

	for (ix=0; ix<width; ++ix)
	{
		cv::Vec3d s(0.0, 0.0, 0.0);
		verSum[0][ix] = s;
		for (iy=0; iy<height; ++iy)
		{
			s += img[iy][ix];
			verSum[iy+1][ix] = s;
		}
	}

	cv::Mat_<cv::Vec3d> aveVerSum(height, width);
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			cv::Vec4b cross = crMap[iy][ix];
			double size = (double)cross[3]+1.0+cross[1];
			aveVerSum[iy][ix][0] = (verSum[iy+cross[3]+1][ix][0]-verSum[iy-cross[1]][ix][0])/(size);
			aveVerSum[iy][ix][1] = (verSum[iy+cross[3]+1][ix][1]-verSum[iy-cross[1]][ix][1])/(size);
			aveVerSum[iy][ix][2] = (verSum[iy+cross[3]+1][ix][2]-verSum[iy-cross[1]][ix][2])/(size);
		}
	}

	cv::Mat_<cv::Vec3d> gradY(height, width);
	for (ix=0; ix<width; ++ix)
	{
		gradY[0][ix][0] = 0.0;
		gradY[0][ix][1] = 0.0;
		gradY[0][ix][2] = 0.0;
		for (iy=1; iy<height; ++iy)
		{
			gradY[iy][ix] = aveVerSum[iy][ix]-aveVerSum[iy-1][ix];
			//printf("%f\n", gradX[iy][ix][0]);
		}
	}

	cv::Mat_<cv::Vec3d> gradXY;
	cv::add(gradX, gradY, gradXY);

	cv::Mat colorArr[3];
	cv::split(gradXY, colorArr);
	NormalizeDoubleAndShow(colorArr[0], "B_Channel_GradXY");
	NormalizeDoubleAndShow(colorArr[1], "G_Channel_GradXY");
	NormalizeDoubleAndShow(colorArr[2], "R_Channel_GradXY");
}

void CFFilter::FastCrossBasedFilter(const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat &src, cv::Mat &dst)
{
	////CalcTime ct;
	//ct.Start();

	int height, width, iy, ix;
	
	height = crMap.rows;
	width = crMap.cols;

	cv::Mat_<cv::Vec4b> crGrad(height, width);

	// calculate cross grad on dx direction
	for (iy=0; iy<height; ++iy)
	{
		int tmpLt, tmpRt;
		tmpLt = -1;
		tmpRt = 0;
		for (ix=0; ix<width; ++ix)
		{
			crGrad[iy][ix][0] = tmpLt+1-crMap[iy][ix][0];
			tmpLt = crMap[iy][ix][0];
			crGrad[iy][ix][2] = crMap[iy][ix][2]+1-tmpRt;
			tmpRt = crMap[iy][ix][2];
		}
	}


	cv::Mat_<cv::Vec3b> img = src;
	cv::Mat_<cv::Vec3i> crHorSum(height, width);

	// scan and filter on dx direction
	for (iy=0; iy<height; ++iy)
	{
		cv::Mat_<cv::Vec3i> sumBuf(1, width+1);
		sumBuf[0][0] = cv::Vec3i(0, 0, 0);
		int lp, rp, k;
		lp = 0;
		rp = 0;
		for (ix=0; ix<width; ++ix)
		{
			for (k=0; k<crGrad[iy][ix][2]; ++k)
			{
				sumBuf[0][rp+1] = sumBuf[0][rp]+(cv::Vec3i)img[iy][rp];
				++rp;
			}
			lp += crGrad[iy][ix][0];
			crHorSum[iy][ix] = sumBuf[0][rp]-sumBuf[0][lp];
		}
	}

	//ct.End("hor_filter end");

	// transpose the image and crossmap
	cv::Mat_<cv::Vec4b> crMapT;
	//crMapT = crMap.clone();
	crMapT = crMap.t();

	crHorSum = crHorSum.t();
	height = crHorSum.rows;
	width = crHorSum.cols;

	crGrad = crGrad.t();

	//ct.End("transpose");

	// calculate cross grad on dy direction, i.e. current dx direction
	for (iy=0; iy<height; ++iy)
	{
		int tmpLt, tmpRt;
		tmpLt = -1;
		tmpRt = 0;
		for (ix=0; ix<width; ++ix)
		{
			crGrad[iy][ix][1] = tmpLt+1-crMapT[iy][ix][1];
			tmpLt = crMapT[iy][ix][1];
			crGrad[iy][ix][3] = crMapT[iy][ix][3]+1-tmpRt;
			tmpRt = crMapT[iy][ix][3];
		}
	}

	// scan and filter on dy direction, i.e. current dx direction
	cv::Mat_<cv::Vec3i> crVerSum(height, width);

	for (iy=0; iy<height; ++iy)
	{
		cv::Mat_<cv::Vec3i> sumBuf(1, width+1);
		sumBuf[0][0] = cv::Vec3i(0, 0, 0);
		int lp, rp, k;
		lp = 0;
		rp = 0;
		for (ix=0; ix<width; ++ix)
		{
			for (k=0; k<crGrad[iy][ix][3]; ++k)
			{
				sumBuf[0][rp+1] = sumBuf[0][rp]+crHorSum[iy][rp];
				++rp;
			}
			lp += crGrad[iy][ix][1];
			crVerSum[iy][ix] = sumBuf[0][rp]-sumBuf[0][lp];
		}
	}
	crVerSum = crVerSum.t();




	//// calculate cross grad on dy direction
	//for (ix=0; ix<width; ++ix)
	//{
	//	int tmpUp, tmpDp;
	//	tmpUp = -1;
	//	tmpDp = 0;
	//	for (iy=0; iy<height; ++iy)
	//	{
	//		crGrad[iy][ix][1] = tmpUp+1-crMap[iy][ix][1];
	//		tmpUp = crMap[iy][ix][1];
	//		crGrad[iy][ix][3] = crMap[iy][ix][3]+1-tmpDp;
	//		tmpDp = crMap[iy][ix][3];
	//	}
	//}

	//// scan and filter on dy direction
	//cv::Mat_<cv::Vec3i> crVerSum(height, width);

	//for (ix=0; ix<width; ++ix)
	//{
	//	cv::Mat_<cv::Vec3i> sumBuf(1, height+1);
	//	sumBuf[0][0] = cv::Vec3i(0, 0, 0);
	//	int up, dp, k;
	//	up = 0;
	//	dp = 0;
	//	for (iy=0; iy<height; ++iy)
	//	{
	//		for (k=0; k<crGrad[iy][ix][3]; ++k)
	//		{
	//			sumBuf[0][dp+1] = sumBuf[0][dp]+crHorSum[dp][ix];
	//			++dp;
	//		}
	//		up += crGrad[iy][ix][1];
	//		crVerSum[iy][ix] = sumBuf[0][dp]-sumBuf[0][up];
	//	}
	//}

	//ct.End("FastCross Filter");

	height = crVerSum.rows;
	width = crVerSum.cols;
	cv::Mat crossSize;
	CountSizeOnCrossMapColMajored(crMap, crossSize);

	dst.create(height, width, CV_8UC3);
	cv::Mat_<cv::Vec3b> res = dst;
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			int pRegion = crossSize.at<ushort>(iy, ix);

			cv::Vec3i inner = crVerSum.at<cv::Vec3i>(iy, ix);

			res[iy][ix][0] = (uchar)(inner[0]/pRegion);
			res[iy][ix][1] = (uchar)(inner[1]/pRegion);
			res[iy][ix][2] = (uchar)(inner[2]/pRegion);

		}
	}
	//ct.End("finish fast filter");
	cv::imshow("Test_Fastfilter", res);
}


void CFFilter::GetCrossGradientUsingSlidingWindow(const cv::Mat_<cv::Vec3b> &img, cv::Mat_<cv::Vec4b> &crGrad, int maxL, int tau)
{
	if ((img.data == NULL) || img.empty()) return;

	//CalcTime ct;
	//ct.Start();

	int width, height;
	width = img.cols;
	height = img.rows;

	crGrad.create(height, width);
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
			int flag = diff;
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

			//crMap[iy][lp][2] = diff-1;
			crGrad[iy][lp][2] = diff-flag;
		}

		int flag = 0;
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

			//crMap[iy][rp][0] = diff-1;
			if (rp < width-1) crGrad[iy][rp+1][0] = diff-flag;
			flag = diff-1;
		}
		crGrad[iy][0][0] = -flag; 
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
			int flag = diff;
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

			//crMap[up][ix][3] = diff-1;
			crGrad[up][ix][3] = diff-flag;
		}

		int flag = 0;
		// to determine up most, cr[1]
		up = height-1;
		for (dp=height-1; dp>=0; --dp)
		{
			int diff = dp-up;
			int flag = diff;
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

			// crMap[dp][ix][1] = diff-1;
			if (dp < height-1) crGrad[dp+1][ix][1] = diff-flag;
			flag = diff-1;
		}
		crGrad[0][ix][1] = -flag; 
	}

	// ct.End("SlWin Construct Cross Gradient");
}

void CFFilter::FastDoubleFilterUsingCrossGradient(const cv::Mat_<cv::Vec4b> &crGrad, const cv::Mat &src, cv::Mat &dst)
{
	//CalcTime ct;
	//ct.Start();

	int height, width, iy, ix;

	height = crGrad.rows;
	width = crGrad.cols;
	//height = crMap.rows;
	//width = crMap.cols;

	//cv::Mat_<cv::Vec4b> crGrad(height, width);

	// //calculate cross grad on dx direction
	//for (iy=0; iy<height; ++iy)
	//{
	//	int tmpLt, tmpRt;
	//	tmpLt = -1;
	//	tmpRt = 0;
	//	for (ix=0; ix<width; ++ix)
	//	{
	//		crGrad[iy][ix][0] = tmpLt+1-crMap[iy][ix][0];
	//		tmpLt = crMap[iy][ix][0];
	//		crGrad[iy][ix][2] = crMap[iy][ix][2]+1-tmpRt;
	//		tmpRt = crMap[iy][ix][2];
	//	}
	//}

	cv::Mat_<double> cost = src;
	//cv::Mat_<cv::Vec3b> img = src;
	//cv::Mat_<cv::Vec3i> crHorSum(height, width);
	cv::Mat_<double> crHorSum(height, width);
	cv::Mat_<int> horSizeSum(height, width);

	// scan and filter on dx direction
#pragma omp parallel for private(iy, ix) schedule(guided, 1)
	for (iy=0; iy<height; ++iy)
	{
		double *sumBuf = new double [width+1];
		//cv::Mat_<cv::Vec3i> sumBuf(1, width+1);
		//sumBuf[0][0] = cv::Vec3i(0, 0, 0);
		sumBuf[0] = 0.0;
		int lp, rp, k;
		lp = 0;
		rp = 0;
		for (ix=0; ix<width; ++ix)
		{
			for (k=0; k<crGrad[iy][ix][2]; ++k)
			{
				//sumBuf[0][rp+1] = sumBuf[0][rp]+(cv::Vec3i)img[iy][rp];
				sumBuf[rp+1] = sumBuf[rp]+cost[iy][rp];
				++rp;
			}
			lp += crGrad[iy][ix][0];
			//crHorSum[iy][ix] = sumBuf[0][rp]-sumBuf[0][lp];
			crHorSum[iy][ix] = sumBuf[rp]-sumBuf[lp];
			horSizeSum[iy][ix] = rp-lp;
		}
		delete [] sumBuf;
	}


	//ct.End("hor_filter end");

	//// transpose the image and crossmap
	//cv::Mat_<cv::Vec4b> crMapT;
	////crMapT = crMap.clone();
	//crMapT = crMap.t();

	//crHorSum = crHorSum.t();
	//height = crHorSum.rows;
	//width = crHorSum.cols;

	//crGrad = crGrad.t();

	//ct.End("transpose");

	//// calculate cross grad on dy direction, i.e. current dx direction
	//for (iy=0; iy<height; ++iy)
	//{
	//	int tmpLt, tmpRt;
	//	tmpLt = -1;
	//	tmpRt = 0;
	//	for (ix=0; ix<width; ++ix)
	//	{
	//		crGrad[iy][ix][1] = tmpLt+1-crMapT[iy][ix][1];
	//		tmpLt = crMapT[iy][ix][1];
	//		crGrad[iy][ix][3] = crMapT[iy][ix][3]+1-tmpRt;
	//		tmpRt = crMapT[iy][ix][3];
	//	}
	//}

	//// scan and filter on dy direction, i.e. current dx direction
	//cv::Mat_<cv::Vec3i> crVerSum(height, width);

	//for (iy=0; iy<height; ++iy)
	//{
	//	cv::Mat_<cv::Vec3i> sumBuf(1, width+1);
	//	sumBuf[0][0] = cv::Vec3i(0, 0, 0);
	//	int lp, rp, k;
	//	lp = 0;
	//	rp = 0;
	//	for (ix=0; ix<width; ++ix)
	//	{
	//		for (k=0; k<crGrad[iy][ix][3]; ++k)
	//		{
	//			sumBuf[0][rp+1] = sumBuf[0][rp]+crHorSum[iy][rp];
	//			++rp;
	//		}
	//		lp += crGrad[iy][ix][1];
	//		crVerSum[iy][ix] = sumBuf[0][rp]-sumBuf[0][lp];
	//	}
	//}
	//crVerSum = crVerSum.t();




	//// calculate cross grad on dy direction
	//for (ix=0; ix<width; ++ix)
	//{
	//	int tmpUp, tmpDp;
	//	tmpUp = -1;
	//	tmpDp = 0;
	//	for (iy=0; iy<height; ++iy)
	//	{
	//		crGrad[iy][ix][1] = tmpUp+1-crMap[iy][ix][1];
	//		tmpUp = crMap[iy][ix][1];
	//		crGrad[iy][ix][3] = crMap[iy][ix][3]+1-tmpDp;
	//		tmpDp = crMap[iy][ix][3];
	//	}
	//}

	// scan and filter on dy direction
	// cv::Mat_<cv::Vec3i> crVerSum(height, width);
	cv::Mat_<double> crVerSum(height, width);
	cv::Mat_<int> verSizeSum(height, width);

	//crHorSum.copyTo(crVerSum);
	//horSizeSum.copyTo(verSizeSum);
#pragma omp parallel for private(iy, ix) schedule(guided, 1)
	for (ix=0; ix<width; ++ix)
	{
		// cv::Mat_<cv::Vec3i> sumBuf(1, height+1);
		double *sumBuf = new double [height+1];
		int *sizeSumBuf = new int [height+1];
		// sumBuf[0][0] = cv::Vec3i(0, 0, 0);
		sumBuf[0] = 0.0;
		sizeSumBuf[0] = 0;
		int up, dp, k;
		up = 0;
		dp = 0;
		for (iy=0; iy<height; ++iy)
		{
			for (k=0; k<crGrad[iy][ix][3]; ++k)
			{
				// sumBuf[0][dp+1] = sumBuf[0][dp]+crHorSum[dp][ix];
				sumBuf[dp+1] = sumBuf[dp]+crHorSum[dp][ix];
				sizeSumBuf[dp+1] = sizeSumBuf[dp]+horSizeSum[dp][ix];
				++dp;
			}
			up += crGrad[iy][ix][1];
			//crVerSum[iy][ix] = sumBuf[0][dp]-sumBuf[0][up];
			crVerSum[iy][ix] = sumBuf[dp]-sumBuf[up];
			verSizeSum[iy][ix] = sizeSumBuf[dp]-sizeSumBuf[up];
		}
		delete [] sumBuf;
		delete [] sizeSumBuf;
	}

	//ct.End("FastCross Filter");

	//height = crVerSum.rows;
	//width = crVerSum.cols;
	//cv::Mat crossSize;
	//CountSizeOnCrossMapColMajored(crMap, crossSize);

	//dst.create(height, width, CV_8UC3);
	dst.create(height, width, CV_64FC1);
	//cv::Mat_<cv::Vec3b> res = dst;
	cv::Mat_<double> res = dst;
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			//int pRegion = crossSize.at<ushort>(iy, ix);
			// int pRegion = verSizeSum[iy][ix];

			//cv::Vec3i inner = crVerSum.at<cv::Vec3i>(iy, ix);
			res[iy][ix] = crVerSum[iy][ix]/verSizeSum[iy][ix];

			/*res[iy][ix][0] = (uchar)(inner[0]/pRegion);
			res[iy][ix][1] = (uchar)(inner[1]/pRegion);
			res[iy][ix][2] = (uchar)(inner[2]/pRegion);*/

		}
	}
	// ct.End("Finish fast filter");
	// cv::imshow("Test_Fastfilter", res);
}

void CFFilter::FastCLMF0DoubleFilter( const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat_<double> &src, cv::Mat_<double> &dst )
{
	int iy, ix, width, height;
	width = crMap.cols;
	height = crMap.rows;

	cv::Mat_<double> cost = src;

	// first iteration
	cv::Mat_<double> crossHorSum(height, width);
	cv::Mat_<int> sizeHorSum(height, width);

	double *horSum = new double [width+1];
	double *rowSizeHorSum = new double [width+1];
	double *verSum = new double [height+1];
	int *colSizeVerSum = new int [height+1];

	for (iy=0; iy<height; ++iy)
	{
		double s = 0.0;
		//double *horSum = new double [width+1];
		// cv::Mat_<double> horSum(1, width+1);
		// horSum[0][0] = s;
		horSum[0] = s;
		for (ix=0; ix<width; ++ix)
		{
			s += cost[iy][ix];
			// horSum[0][ix+1] = s;
			horSum[ix+1] = s;
		}

		for (ix=0; ix<width; ++ix)
		{
			cv::Vec4b cross = crMap[iy][ix];
			// crossHorSum[iy][ix] = horSum[0][ix+cross[2]+1]-horSum[0][ix-cross[0]];
			crossHorSum[iy][ix] = horSum[ix+cross[2]+1]-horSum[ix-cross[0]];
			sizeHorSum[iy][ix] = cross[2]+cross[0]+1;
		}
		//delete [] horSum;
	}


	cv::Mat_<double> crossVerSum(height, width);
	cv::Mat_<int> sizeVerSum(height, width);

	for (ix=0; ix<width; ++ix)
	{
		//cv::Mat_<double> verSum(height+1, 1);
		//cv::Mat_<int> colSizeVerSum(height+1, 1);
		// cv::Mat_<double> verSum(1, height+1);
		// cv::Mat_<int> colSizeVerSum(1, height+1);
		double s = 0.0;
		int cs = 0;	
		//double *verSum = new double [height+1];
		//int *colSizeVerSum = new int [height+1];
		// verSum[0][0] = s;
		// colSizeVerSum[0][0] = cs;
		verSum[0] = s;
		colSizeVerSum[0] = cs;
		for (iy=0; iy<height; ++iy)
		{
			s += crossHorSum[iy][ix];
			// verSum[0][iy+1] = s;
			verSum[iy+1] = s;
			cs += sizeHorSum[iy][ix];
			// colSizeVerSum[0][iy+1] = cs;
			colSizeVerSum[iy+1] = cs;
		}

		for (iy=0; iy<height; ++iy)
		{
			cv::Vec4b cross = crMap[iy][ix];
			/*crossVerSum[iy][ix] = verSum[0][iy+cross[3]+1]-verSum[0][iy-cross[1]];
			sizeVerSum[iy][ix] = colSizeVerSum[0][iy+cross[3]+1]-colSizeVerSum[0][iy-cross[1]];*/
			crossVerSum[iy][ix] = verSum[iy+cross[3]+1]-verSum[iy-cross[1]];
			sizeVerSum[iy][ix] = colSizeVerSum[iy+cross[3]+1]-colSizeVerSum[iy-cross[1]];
		}

		//delete [] verSum;
		//delete [] colSizeVerSum;
	}


	// second iteration
	// cv::Mat_<double> crossHorSum(height, width);
	// cv::Mat_<int> sizeHorSum(height, width);
	//double *rowSizeHorSum = new double [width+1];
	for (iy=0; iy<height; ++iy)
	{
		double s = 0.0;
		// cv::Mat_<double> horSum(1, width+1);
		//double *horSum = new double [width+1];
		//double *rowSizeHorSum = new double [width+1];
		//horSum[0][0] = s;
		horSum[0] = s;
		int rs = 0;
		// cv::Mat_<int> rowSizeHorSum(1, width+1);
		
		//rowSizeHorSum[0][0] = rs;
		rowSizeHorSum[0] = rs;
		for (ix=0; ix<width; ++ix)
		{
			s += crossVerSum[iy][ix];
			// horSum[0][ix+1] = s;
			horSum[ix+1] = s;
			rs += sizeVerSum[iy][ix];
			// rowSizeHorSum[0][ix+1] = rs;
			rowSizeHorSum[ix+1] = rs;
		}

		for (ix=0; ix<width; ++ix)
		{
			cv::Vec4b cross = crMap[iy][ix];
			/*crossHorSum[iy][ix] = horSum[0][ix+cross[2]+1]-horSum[0][ix-cross[0]];
			sizeHorSum[iy][ix] = rowSizeHorSum[0][ix+cross[2]+1]-rowSizeHorSum[0][ix-cross[0]];*/
			crossHorSum[iy][ix] = horSum[ix+cross[2]+1]-horSum[ix-cross[0]];
			sizeHorSum[iy][ix] = rowSizeHorSum[ix+cross[2]+1]-rowSizeHorSum[ix-cross[0]];
		}
		//delete [] horSum;
		//delete [] rowSizeHorSum;

	}

	//cv::Mat_<double> crossVerSum(height, width);
	//cv::Mat_<int> sizeVerSum(height, width);
	for (ix=0; ix<width; ++ix)
	{
		//cv::Mat_<double> verSum(height+1, 1);
		//cv::Mat_<int> colSizeVerSum(height+1, 1);
		// cv::Mat_<double> verSum(1, height+1);
		// cv::Mat_<int> colSizeVerSum(1, height+1);
		//double *verSum = new double [height+1];
		//int *colSizeVerSum = new int [height+1];
		double s = 0.0;
		int cs = 0;
		//verSum[0][0] = s;
		//colSizeVerSum[0][0] = cs;
		verSum[0] = s;
		colSizeVerSum[0] = cs;
		for (iy=0; iy<height; ++iy)
		{
			s += crossHorSum[iy][ix];
			// verSum[0][iy+1] = s;
			verSum[iy+1] = s;
			
			cs += sizeHorSum[iy][ix];
			// colSizeVerSum[0][iy+1] = cs;
			colSizeVerSum[iy+1] = cs;
		}

		for (iy=0; iy<height; ++iy)
		{
			cv::Vec4b cross = crMap[iy][ix];
			/*crossVerSum[iy][ix] = verSum[0][iy+cross[3]+1]-verSum[0][iy-cross[1]];
			sizeVerSum[iy][ix] = colSizeVerSum[0][iy+cross[3]+1]-colSizeVerSum[0][iy-cross[1]];*/
			crossVerSum[iy][ix] = verSum[iy+cross[3]+1]-verSum[iy-cross[1]];
			sizeVerSum[iy][ix] = colSizeVerSum[iy+cross[3]+1]-colSizeVerSum[iy-cross[1]];
		}
		//delete [] verSum;
		//delete [] colSizeVerSum;

	}

	delete [] horSum;
	delete [] rowSizeHorSum;
	delete [] verSum;
	delete [] colSizeVerSum;

	dst.create(height, width);
	cv::Mat_<double> tmpDst = dst;
	double *pCrossVerSum = (double *)crossVerSum.ptr(0);
	int *pSizeVerSum = (int *)sizeVerSum.ptr(0);
	double *pTmpDst = (double *)tmpDst.ptr(0);
	//for (iy=0; iy<height; ++iy)
	for (iy=0; iy<height*width; ++iy)
	{
		//for (ix=0; ix<width; ++ix)
		{
			//tmpDst[iy][ix] = crossVerSum[iy][ix]/sizeVerSum[iy][ix];
			(*pTmpDst++) = (*pCrossVerSum++)/(*pSizeVerSum++);
		} 
	}
}


void CFFilter::FastCLMF0FloatFilter( const cv::Mat_<cv::Vec4b> &crMap, const cv::Mat_<float> &src, cv::Mat_<float> &dst )
{
	int iy, ix, width, height;
	width = crMap.cols;
	height = crMap.rows;

	cv::Mat_<float> cost = src;

	// first iteration
	cv::Mat_<float> crossHorSum(height, width);
	cv::Mat_<int> sizeHorSum(height, width);

	float *horSum = new float [width+1];
	float *rowSizeHorSum = new float [width+1];
	float *verSum = new float [height+1];
	int *colSizeVerSum = new int [height+1];

	for (iy=0; iy<height; ++iy)
	{
		float s = 0.0;
		horSum[0] = s;
		for (ix=0; ix<width; ++ix)
		{
			s += cost[iy][ix];
			horSum[ix+1] = s;
		}

		for (ix=0; ix<width; ++ix)
		{
			cv::Vec4b cross = crMap[iy][ix];
			crossHorSum[iy][ix] = horSum[ix+cross[2]+1]-horSum[ix-cross[0]];
			sizeHorSum[iy][ix] = cross[2]+cross[0]+1;
		}
	}


	cv::Mat_<float> crossVerSum(height, width);
	cv::Mat_<int> sizeVerSum(height, width);

	for (ix=0; ix<width; ++ix)
	{
		float s = 0.0;
		int cs = 0;	
		verSum[0] = s;
		colSizeVerSum[0] = cs;
		for (iy=0; iy<height; ++iy)
		{
			s += crossHorSum[iy][ix];
			verSum[iy+1] = s;
			cs += sizeHorSum[iy][ix];
			colSizeVerSum[iy+1] = cs;
		}

		for (iy=0; iy<height; ++iy)
		{
			cv::Vec4b cross = crMap[iy][ix];
			crossVerSum[iy][ix] = verSum[iy+cross[3]+1]-verSum[iy-cross[1]];
			sizeVerSum[iy][ix] = colSizeVerSum[iy+cross[3]+1]-colSizeVerSum[iy-cross[1]];
		}
	}


	// second iteration
	for (iy=0; iy<height; ++iy)
	{
		float s = 0.0;
		horSum[0] = s;
		int rs = 0;
		rowSizeHorSum[0] = rs;
		for (ix=0; ix<width; ++ix)
		{
			s += crossVerSum[iy][ix];
			horSum[ix+1] = s;
			rs += sizeVerSum[iy][ix];
			rowSizeHorSum[ix+1] = rs;
		}

		for (ix=0; ix<width; ++ix)
		{
			cv::Vec4b cross = crMap[iy][ix];
			crossHorSum[iy][ix] = horSum[ix+cross[2]+1]-horSum[ix-cross[0]];
			sizeHorSum[iy][ix] = rowSizeHorSum[ix+cross[2]+1]-rowSizeHorSum[ix-cross[0]];
		}
	}

	for (ix=0; ix<width; ++ix)
	{
		float s = 0.0;
		int cs = 0;
		verSum[0] = s;
		colSizeVerSum[0] = cs;
		for (iy=0; iy<height; ++iy)
		{
			s += crossHorSum[iy][ix];
			verSum[iy+1] = s;
			
			cs += sizeHorSum[iy][ix];
			colSizeVerSum[iy+1] = cs;
		}

		for (iy=0; iy<height; ++iy)
		{
			cv::Vec4b cross = crMap[iy][ix];
			crossVerSum[iy][ix] = verSum[iy+cross[3]+1]-verSum[iy-cross[1]];
			sizeVerSum[iy][ix] = colSizeVerSum[iy+cross[3]+1]-colSizeVerSum[iy-cross[1]];
		}

	}

	delete [] horSum;
	delete [] rowSizeHorSum;
	delete [] verSum;
	delete [] colSizeVerSum;

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

/*
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

	//float** crossHorSum = qx_allocf(height, width);
	//int** sizeHorSum = qx_alloci(height, width);


//	float *horSum = (float*)malloc(sizeof(float)*(width+1));
//	float *rowSizeHorSum = (float*)malloc(sizeof(float)*(width+1));
//	float *verSum = (float*)malloc(sizeof(float)*(height+1));
//	int *colSizeVerSum = (int*)malloc(sizeof(int)*(height+1));


	float *horSum = new float [width+1];
	float *rowSizeHorSum = new float [width+1];
	float *verSum = new float [height+1];
	int *colSizeVerSum = new int [height+1];

	float *costPtr = (float *)(cost.ptr(0));
//	float *crossHorPtr = (float *)(crossHorSum.ptr(0));
//	int *sizeHorPtr = (int *)(sizeHorSum.ptr(0));
	float *crossHorPtr = crossHorSum[0];
	int *sizeHorPtr = sizeHorSum[0];
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
			cv::Vec4b cross = crMap[iy][ix];
			*crossHorPtr++ = horSum[ix+cross[2]+1]-horSum[ix-cross[0]];
			*sizeHorPtr++ = cross[2]+cross[0]+1;
		}
	}

	//float** crossVerSum = qx_allocf(height, width);
	//int** sizeVerSum = qx_alloci(height, width);
	cv::Mat_<float> crossVerSum(height, width);
	cv::Mat_<int> sizeVerSum(height, width);
	const int W_FAC = width;
	for (ix=0; ix<width; ++ix)
	{
		float s = 0.0;
		int cs = 0;	
		
		float *crossHorPtr = (float *)(crossHorSum.ptr(0))+ix;
		int *sizeHorPtr = (int *)(sizeHorSum.ptr(0))+ix;
		//float *crossHorPtr = &crossHorSum[0][ix];
		//int *sizeHorPtr = &sizeHorSum[0][ix];

		float *verPtr = verSum;
		int *colSizeVerPtr = colSizeVerSum;
		*verPtr++ = s;
		*colSizeVerPtr++ = cs;
		for (iy=0; iy<height; ++iy, crossHorPtr+=W_FAC, sizeHorPtr+=W_FAC)
		{
			s += *crossHorPtr;
			*verPtr++ =s;
			cs += *sizeHorPtr;
			*colSizeVerPtr++ = cs;
		}

		float *crossVerPtr = (float *)(crossVerSum.ptr(0))+ix;
		int *sizeVerPtr = (int *)(sizeVerSum.ptr(0))+ix;
		//float *crossVerPtr = &crossVerSum[0][ix];
		//int *sizeVerPtr = &sizeVerSum[0][ix];

		for (iy=0; iy<height; ++iy, crossVerPtr+=W_FAC, sizeVerPtr+=W_FAC)
		{
			cv::Vec4b cross = crMap[iy][ix];
			*crossVerPtr = verSum[iy+cross[3]+1]-verSum[iy-cross[1]];
			*sizeVerPtr = colSizeVerSum[iy+cross[3]+1]-colSizeVerSum[iy-cross[1]];
		}
	}


	// second iteration
	float *crossVerPtr = (float *)(crossVerSum.ptr(0));
	int *sizeVerPtr = (int *)(sizeVerSum.ptr(0));
	crossHorPtr = (float *)(crossHorSum.ptr(0));
	sizeHorPtr = (int *)(sizeHorSum.ptr(0));
//	float *crossVerPtr = crossVerSum[0];
//	int *sizeVerPtr = sizeVerSum[0];
//	crossHorPtr = crossHorSum[0];
//	sizeHorPtr = sizeHorSum[0];
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
			cv::Vec4b cross = crMap[iy][ix];
			*crossHorPtr++ = horSum[ix+cross[2]+1]-horSum[ix-cross[0]];
			*sizeHorPtr++ = rowSizeHorSum[ix+cross[2]+1]-rowSizeHorSum[ix-cross[0]];
		}
	}

	for (ix=0; ix<width; ++ix)
	{
		float s = 0.0;
		int cs = 0;
		float *verPtr = verSum;
		int *colSizeVerPtr = colSizeVerSum;
		*verPtr++ = s;
		*colSizeVerPtr++ = cs;

		float *crossHorPtr = (float *)(crossHorSum.ptr(0))+ix;
		int *sizeHorPtr = (int *)(sizeHorSum.ptr(0))+ix;
		//float *crossHorPtr = &crossHorSum[0][ix];
		//int *sizeHorPtr = &sizeHorSum[0][ix];
		for (iy=0; iy<height; ++iy, crossHorPtr+=W_FAC, sizeHorPtr+=W_FAC)
		{
			s += *crossHorPtr;
			*verPtr++ = s;
			cs += *sizeHorPtr;
			*colSizeVerPtr++ = cs;
		}

		float *crossVerPtr = (float *)(crossVerSum.ptr(0))+ix;
		int *sizeVerPtr = (int *)(sizeVerSum.ptr(0))+ix;
		//float *crossVerPtr = &crossVerSum[0][ix];
		//int *sizeVerPtr = &sizeVerSum[0][ix];
		for (iy=0; iy<height; ++iy, crossVerPtr+=W_FAC, sizeVerPtr+=W_FAC)
		{
			cv::Vec4b cross = crMap[iy][ix];
			*crossVerPtr = verSum[iy+cross[3]+1]-verSum[iy-cross[1]];
			*sizeVerPtr = colSizeVerSum[iy+cross[3]+1]-colSizeVerSum[iy-cross[1]];
		}

	}
	//tt.time_display("Smoothing");

	delete [] horSum;
	delete [] rowSizeHorSum;
	delete [] verSum;
	delete [] colSizeVerSum;

	dst.create(height, width);
	cv::Mat_<float> tmpDst = dst;
	float *pCrossVerSum = (float *)crossVerSum.ptr(0);
	int *pSizeVerSum = (int *)sizeVerSum.ptr(0);
	//float *pCrossVerSum = crossVerSum[0];
	//int *pSizeVerSum = sizeVerSum[0];
	float *pTmpDst = (float *)tmpDst.ptr(0);

	for (iy=0; iy<height*width; ++iy)
	{
		(*pTmpDst++) = (*pCrossVerSum++)/(*pSizeVerSum++);
	}
	
//	free(horSum);
//	free(rowSizeHorSum);
//	free(verSum);
//	free(colSizeVerSum);
//	qx_freef(crossHorSum);
//	qx_freei(sizeHorSum);
//	qx_freef(crossVerSum);
//	qx_freei(sizeVerSum);
}*/
#pragma region UGLY_CODE_FROM_STEVEN
/*************************************************
Function:       // Skeleton5Build
Description:    // build up support window for color image

Others:		You can control all these factors to achieve better performance

			0. Using two \tau and two \L (L2 & HNum), Threshold1 for length < L2, Threshold 2 otherwise..(L2<length<HNum)
			1. Using moving average
				a.new element is assigned weight with 1/5
			2. No Median Filter before building step
			3. No Using next neighbor to check current neighbor with same scheme as 1
			4. No symmetric scheme for single image

*************************************************/
void Skeleton5Build(int **pImgCpy, int HSet[], int HNum, int ImgWidth, int ImgHeight, int **pWin)
{    
	int iy, ix, PixIdx, PixIdxNb, ih, Scale, OptScale;

	//tune threshold
	int Threshold;
	int Threshold1 = 25;
	//--tau adjustment
	int Threshold2 = 15;   
	int L2 = 5;

	int iyy;

	int *centerNewValue = new int[3];
	int channel;
	int *tempNewValue = new int[3];

	//MedianFilterImage(pImgCpy, ImgWidth, ImgHeight);

	for(iy = 0; iy < ImgHeight; iy++)
	{
		iyy = iy * ImgWidth;
		for(ix=0; ix<ImgWidth; ix++)
		{
			PixIdx = iyy + ix;

			//Direction >
			Threshold = Threshold1;
			/*for(channel = 0; channel < 3; channel++)				//initialize the anchor point intensity
			{
				centerNewValue[channel] = pImgCpy[PixIdx][channel];
			}*/

			OptScale = HSet[0];
			if(OptScale + ix >= ImgWidth) //right border, to set right arm to 0
			{
				OptScale = ImgWidth - 1 - ix;  
			}
			/*
			for(channel = 0; channel < 3; channel++)		//add the next one intensity and average
			{
				centerNewValue[channel] = (centerNewValue[channel] + pImgCpy[PixIdx + OptScale][channel]) / 2;
			}*/
			for(channel = 0; channel < 3; channel++)		//add the next one intensity and average
			{
				centerNewValue[channel] = (pImgCpy[PixIdx][channel] + pImgCpy[PixIdx + OptScale][channel]) / 2;
			}

			for(ih=1; ih<HNum; ih++)
			{
				Scale = HSet[ih];
				if(Scale > L2)
					Threshold = Threshold2;
				PixIdxNb = PixIdx+Scale;
				if(ix+ Scale < ImgWidth)
				{
					if(																//pixel itself is not friend, compared with new center
						(abs(pImgCpy[PixIdxNb][0] - centerNewValue[0])  > Threshold)
						||(abs(pImgCpy[PixIdxNb][1] - centerNewValue[1]) > Threshold)
						||(abs(pImgCpy[PixIdxNb][2] - centerNewValue[2]) > Threshold)/*
						(abs(pImgCpy[PixIdxNb][0] - pImgCpy[PixIdx][0])  > Threshold)
						||(abs(pImgCpy[PixIdxNb][1] - pImgCpy[PixIdx][1]) > Threshold)
						||(abs(pImgCpy[PixIdxNb][2] - pImgCpy[PixIdx][2]) > Threshold)*/
						)
					{		
						/*
						//----------------------
						//enhanced cross building approach
						if((ix + Scale + 1) >= ImgWidth)	//?maybe the last point we need to include without checking
						{
							OptScale = Scale;
							break;								//right border
						}
						else if((ix + Scale + 1) < ImgWidth)	
						{
							if(																	//pixel its right one is not friend
								(abs(pImgCpy[PixIdxNb + 1][0] -centerNewValue[0])  > Threshold)
								||(abs(pImgCpy[PixIdxNb + 1][1] - centerNewValue[1]) > Threshold)
								||(abs(pImgCpy[PixIdxNb + 1][2] - centerNewValue[2]) > Threshold)
								)
								break;
							else																//pixel its right one is friend, not stopped
							{
								OptScale = Scale;
								for(channel = 0; channel < 3; channel++)
								{
									tempNewValue[channel] = centerNewValue[channel];			//if an noise point occur, this point intensity will not be added into average one
								}
							}
						}
						//over
						//----------------------------------
						*/
						break; 
					}
					else				
					{					
						/*for(channel = 0; channel < 3; channel++)					//if point is friend, add itself
						{
							tempNewValue[channel] = pImgCpy[PixIdxNb][channel];
						}*/
						/*for(channel = 0; channel < 3; channel++)					//tune the anchor point anchor by average this
						{
							centerNewValue[channel] = (centerNewValue[channel] * (ih + 1) + pImgCpy[PixIdxNb][channel]) / (ih + 2);
						}*/
						for(channel = 0; channel < 3; channel++)					//tune the anchor point anchor by average this
						{
							centerNewValue[channel] = (centerNewValue[channel] * 4 + pImgCpy[PixIdxNb][channel]) / 5;
						}
						OptScale = Scale;	//the pixel itself is friend
					}
				}
				else
				{
					break;
				}
				/*
				for(channel = 0; channel < 3; channel++)					//tune the anchor point anchor by average this
				{
					centerNewValue[channel] = (centerNewValue[channel] * (ih + 1) + tempNewValue[channel]) / (ih + 2);
				}*/
			}
			pWin[PixIdx][0] = OptScale;

			//Direction ^
			Threshold = Threshold1;/*
			for(channel = 0; channel < 3; channel++)				//initialize the anchor point intensity
			{
				centerNewValue[channel] = pImgCpy[PixIdx][channel];
			}*/

			OptScale = HSet[0];
			if(iy - OptScale< 0) //top boarder
			{
				OptScale = iy;  
			}
			/*
			for(channel = 0; channel < 3; channel++)		//add the next one intensity and average
			{

				centerNewValue[channel] = (centerNewValue[channel] + pImgCpy[PixIdx - OptScale * ImgWidth][channel]) / 2;
			}*/
			for(channel = 0; channel < 3; channel++)		//add the next one intensity and average
			{
				centerNewValue[channel] = (pImgCpy[PixIdx][channel] + pImgCpy[PixIdx - OptScale * ImgWidth][channel]) / 2;
			}

			for(ih=1; ih<HNum; ih++)
			{
				Scale = HSet[ih];
				if (Scale > L2)
				{
					Threshold = Threshold2;
				}
				PixIdxNb = PixIdx - Scale*ImgWidth;
				if(iy - Scale >= 0)
				{
					if(																//pixel itself is not friend
						(abs(pImgCpy[PixIdxNb][0] - centerNewValue[0]) > Threshold)
						||(abs(pImgCpy[PixIdxNb][1] - centerNewValue[1]) > Threshold)
						||(abs(pImgCpy[PixIdxNb][2] - centerNewValue[2]) > Threshold)/*
						(abs(pImgCpy[PixIdxNb][0] - pImgCpy[PixIdx][0])  > Threshold)
						||(abs(pImgCpy[PixIdxNb][1] - pImgCpy[PixIdx][1]) > Threshold)
						||(abs(pImgCpy[PixIdxNb][2] - pImgCpy[PixIdx][2]) > Threshold)*/
						)
					{
						/*
						//----------------------<
						//enhanced cross building approach
						if((iy - Scale - 1) < 0)		//top border
						{
							OptScale = Scale;	
							break;
						}
						else if((iy - Scale - 1) >= 0)	
						{
							if(																	//pixel its upper one is not friend
								(abs(pImgCpy[PixIdxNb - ImgWidth][0] - centerNewValue[0])  > Threshold)
								||(abs(pImgCpy[PixIdxNb - ImgWidth][1] - centerNewValue[1]) > Threshold)
								||(abs(pImgCpy[PixIdxNb - ImgWidth][2] - centerNewValue[2]) > Threshold)
								)
								break;
							else																//pixel its upper one is friend, not stopped
							{
								for(channel = 0; channel < 3; channel++)
								{
									tempNewValue[channel] = centerNewValue[channel];			//if an noise point occur, this point intensity will not be added into average one
								}
								OptScale = Scale;						 
							}
						}
						//over
						//---------------------------------->
						*/
						break; 
					}
					else
					{	/*
						for(channel = 0; channel < 3; channel++)					//if point is friend, i
						{
							tempNewValue[channel] = pImgCpy[PixIdxNb][channel];
						}*/
						/*for(channel = 0; channel < 3; channel++)					//tune the anchor point anchor by average this
						{
							centerNewValue[channel] = (centerNewValue[channel] * (ih + 1) + pImgCpy[PixIdxNb][channel]) / (ih + 2);
						}*/
						for(channel = 0; channel < 3; channel++)					//tune the anchor point anchor by average this
						{
							centerNewValue[channel] = (centerNewValue[channel] * 4 + pImgCpy[PixIdxNb][channel]) / 5;
						}
						OptScale = Scale;		//pixel itself is friend
					}
				}
				else
				{
					break;
				}
				/*
				for(channel = 0; channel < 3; channel++)					//tune the anchor point anchor by average this
				{
					centerNewValue[channel] = (centerNewValue[channel] * (ih + 1) + tempNewValue[channel]) / (ih + 2);
				}*/
			}
			pWin[PixIdx][1] = OptScale;

			//Direction <
			Threshold = Threshold1;/*
			for(channel = 0; channel < 3; channel++)				//initialize the anchor point intensity
			{
				centerNewValue[channel] = pImgCpy[PixIdx][channel];
			}*/

			OptScale = HSet[0];
			if(ix - OptScale< 0) //left boarder
			{
				OptScale = ix;  
			}
			/*
			for(channel = 0; channel < 3; channel++)		//add the next one intensity and average
			{
				centerNewValue[channel] = (centerNewValue[channel] + pImgCpy[PixIdx - OptScale][channel]) / 2;
			}*/
			for(channel = 0; channel < 3; channel++)		//add the next one intensity and average
			{
				centerNewValue[channel] = (pImgCpy[PixIdx][channel] + pImgCpy[PixIdx - OptScale][channel]) / 2;
			}

			for(ih=1; ih<HNum; ih++)
			{
				Scale = HSet[ih];
				if (Scale > L2)
					Threshold = Threshold2;
				PixIdxNb = PixIdx - Scale;
				if(ix- Scale >= 0)
				{
					if(																//pixel itself is not friend
						(abs(pImgCpy[PixIdxNb][0] - centerNewValue[0]) > Threshold)
						||(abs(pImgCpy[PixIdxNb][1] - centerNewValue[1]) > Threshold)
						||(abs(pImgCpy[PixIdxNb][2] - centerNewValue[2]) > Threshold)/*
						(abs(pImgCpy[PixIdxNb][0] - pImgCpy[PixIdx][0])  > Threshold)
						||(abs(pImgCpy[PixIdxNb][1] - pImgCpy[PixIdx][1]) > Threshold)
						||(abs(pImgCpy[PixIdxNb][2] - pImgCpy[PixIdx][2]) > Threshold)*/
						)
					{
						/*
						//----------------------<
						//enhanced cross building approach
						if((ix - Scale - 1) < 0)	
						{
							OptScale = Scale;
							break;								//left border
						}
						else if((ix - Scale - 1) >= 0)			
						{
							if(																	//pixel its left one is not friend
								(abs(pImgCpy[PixIdxNb - 1][0] - centerNewValue[0])  > Threshold)
								||(abs(pImgCpy[PixIdxNb - 1][1] - centerNewValue[1]) > Threshold)
								||(abs(pImgCpy[PixIdxNb - 1][2] - centerNewValue[2]) > Threshold)
								)
								break;
							else																//pixel its left one is friend, not stopped
							{
								for(channel = 0; channel < 3; channel++)
								{
									tempNewValue[channel] = centerNewValue[channel];			//if an noise point occur, this point intensity will not be added into average one
								}
								OptScale = Scale;
							}
						}
						//over
						//---------------------------------->
						*/
						break; 
					}
					else
					{	/*
						for(channel = 0; channel < 3; channel++)					//if point is friend, i
						{
							tempNewValue[channel] = pImgCpy[PixIdxNb][channel];
						}*/
						/*for(channel = 0; channel < 3; channel++)					//tune the anchor point anchor by average this
						{
							centerNewValue[channel] = (centerNewValue[channel] * (ih + 1) + pImgCpy[PixIdxNb][channel]) / (ih + 2);
						}*/
						for(channel = 0; channel < 3; channel++)					//tune the anchor point anchor by average this
						{
							centerNewValue[channel] = (centerNewValue[channel] * 4 + pImgCpy[PixIdxNb][channel]) / 5;
						}
						OptScale = Scale;	//pixel itself is friend
					}
				}
				else
				{
					break;
				}/*
				for(channel = 0; channel < 3; channel++)					//tune the anchor point anchor by average this
				{
					centerNewValue[channel] = (centerNewValue[channel] * (ih + 1) + tempNewValue[channel]) / (ih + 2);
				}*/
			}
			pWin[PixIdx][2] = OptScale;


			//Direction 
			Threshold = Threshold1;/*
			for(channel = 0; channel < 3; channel++)				//initialize the anchor point intensity
			{
				centerNewValue[channel] = pImgCpy[PixIdx][channel];
			}*/

			OptScale = HSet[0];
			if(iy + OptScale>= ImgHeight) //bottom boarder
			{
				OptScale = ImgHeight - 1 - iy;  
			}/*
			for(channel = 0; channel < 3; channel++)		//add the next one intensity and average
			{
				centerNewValue[channel] = (centerNewValue[channel] + pImgCpy[PixIdx + OptScale * ImgWidth][channel]) / 2;
			}*/
			for(channel = 0; channel < 3; channel++)		//add the next one intensity and average
			{
				centerNewValue[channel] = (pImgCpy[PixIdx][channel] + pImgCpy[PixIdx + OptScale * ImgWidth][channel]) / 2;
			}
			for(ih=1; ih<HNum; ih++)
			{
				Scale = HSet[ih];
				if(Scale > L2)
					Threshold = Threshold2;
				PixIdxNb = PixIdx+Scale*ImgWidth;	
				if(iy + Scale < ImgHeight)
				{
					if(																//pixel itself is not friend
						(abs(pImgCpy[PixIdxNb][0] - centerNewValue[0]) > Threshold)
						||(abs(pImgCpy[PixIdxNb][1] - centerNewValue[1]) > Threshold)
						||(abs(pImgCpy[PixIdxNb][2] - centerNewValue[2]) > Threshold)/*
						(abs(pImgCpy[PixIdxNb][0] - pImgCpy[PixIdx][0])  > Threshold)
						||(abs(pImgCpy[PixIdxNb][1] - pImgCpy[PixIdx][1]) > Threshold)
						||(abs(pImgCpy[PixIdxNb][2] - pImgCpy[PixIdx][2]) > Threshold)*/
						)
					{
						/*
						//----------------------<
						//enhanced cross building approach
						if((iy + Scale + 1) >= ImgHeight)		//bottom border
						{
							OptScale = Scale;
							break;								
						}
						else if((iy + Scale + 1) < ImgHeight)	
						{
							if(																	//pixel its under one is not friend
								(abs(pImgCpy[PixIdxNb + ImgWidth][0] - centerNewValue[0])  > Threshold)
								||(abs(pImgCpy[PixIdxNb + ImgWidth][1] - centerNewValue[1]) > Threshold)
								||(abs(pImgCpy[PixIdxNb + ImgWidth][2] - centerNewValue[2]) > Threshold)
								)
								break;
							else																//pixel its under one is friend, not stopped
							{
								for(channel = 0; channel < 3; channel++)
								{
									tempNewValue[channel] = centerNewValue[channel];			//if an noise point occur, this point intensity will not be added into average one
								}
								OptScale = Scale;		//pixel itself is friend
							}
						}
						//over
						//---------------------------------->
						*/
						break; 
					}
					else
					{	/*
						for(channel = 0; channel < 3; channel++)					//if point is friend, i
						{
							tempNewValue[channel] = pImgCpy[PixIdxNb][channel];
						}*/
						/*for(channel = 0; channel < 3; channel++)					//tune the anchor point anchor by average this
						{
							centerNewValue[channel] = (centerNewValue[channel] * (ih + 1) + pImgCpy[PixIdxNb][channel]) / (ih + 2);
						}*/
						for(channel = 0; channel < 3; channel++)					//tune the anchor point anchor by average this
						{
							centerNewValue[channel] = (centerNewValue[channel] * 4 + pImgCpy[PixIdxNb][channel]) / 5;
						}
						OptScale = Scale;
					}
				}
				else
				{
					break;
				}/*
				for(channel = 0; channel < 3; channel++)					//tune the anchor point anchor by average this
				{
					centerNewValue[channel] = (centerNewValue[channel] * (ih + 1) + tempNewValue[channel]) / (ih + 2);
				}*/
			}
			pWin[PixIdx][3] = OptScale; 
		}
	}
}

#pragma endregion

void CFFilter::WrapperForSkelonBuild(const cv::Mat_<cv::Vec3b> &img, int armLength, cv::Mat_<cv::Vec4b> &crMap)
{
	int *hSet = new int [armLength];
	int iy, ix, height, width;
	for (iy=0; iy<armLength; ++iy)
		hSet[iy] = iy+1;

	height = img.rows;
	width = img.cols;
	int **pImg = new int* [height*width];
	int **pWin = new int* [height*width];
	for (iy=0; iy<height*width; ++iy) 
	{
		pImg[iy] = new int [3];
		pWin[iy] = new int [4];
	}

	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			int tmpId = iy*width+ix;
			pImg[tmpId][0] = img[iy][ix][0];
			pImg[tmpId][1] = img[iy][ix][1];
			pImg[tmpId][2] = img[iy][ix][2];
		}
	}

	Skeleton5Build(pImg, hSet, armLength, width, height, pWin);

	crMap.create(height, width);
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			int tmpId = iy*width+ix;
			crMap[iy][ix][0] = pWin[tmpId][2];
			crMap[iy][ix][1] = pWin[tmpId][1];
			crMap[iy][ix][2] = pWin[tmpId][0];
			crMap[iy][ix][3] = pWin[tmpId][3];
		}
	}

	delete [] hSet;
	for (iy=0; iy<height*width; ++iy)
	{
		delete [] pImg[iy];
		delete [] pWin[iy];
	}
	delete [] pImg;
	delete [] pWin;
}