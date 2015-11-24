#include "Superpixels_Header.h"
#include "SLIC.h"

#include <vector>
using std::vector;

using std::max;
using std::min;

//=================================================================================
/// [NOTE: extracted and modified from SLIC source code]
/// 
/// DrawContoursAroundSegments
///
/// Internal contour drawing option exists. One only needs to comment the if
/// statement inside the loop that looks at neighbourhood.
//=================================================================================
void DrawContoursAroundSegments(const cv::Mat_<cv::Vec3b> &imgIn, cv::Mat_<int>	&SegLabels, cv::Mat_<cv::Vec3b> &imgOut)
{
	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

	int width = imgIn.cols;
	int height = imgIn.rows;

/*	int sz = width*height;

	vector<bool> istaken(sz, false);

	int mainindex(0);
	for( int j = 0; j < height; j++ )
	{
		for( int k = 0; k < width; k++ )
		{
			int np(0);
			for( int i = 0; i < 8; i++ )
			{
				int x = k + dx8[i];
				int y = j + dy8[i];

				if( (x >= 0 && x < width) && (y >= 0 && y < height) )
				{
					int index = y*width + x;

					if( false == istaken[index] )//comment this to obtain internal contours
					{
						if( labels[mainindex] != labels[index] ) np++;
					}
				}
			}
			if( np > 1 )//change to 2 or 3 for thinner lines
			{
				ubuff[mainindex] = color;
				istaken[mainindex] = true;
			}
			mainindex++;
		}
	}*/

	int *labels = (int *)SegLabels.ptr(0);

	int sz = width*height;
	vector<bool> istaken(sz, false);
	vector<int> contourx(sz);
	vector<int> contoury(sz);
	int mainindex(0);
	int cind(0);
	for( int j = 0; j < height; j++ )
	{
		for( int k = 0; k < width; k++ )
		{
			int np(0);
			for( int i = 0; i < 8; i++ )
			{
				int x = k + dx8[i];
				int y = j + dy8[i];

				if( (x >= 0 && x < width) && (y >= 0 && y < height) )
				{
					int index = y*width + x;

					//if( false == istaken[index] )//comment this to obtain internal contours
					{
						if( labels[mainindex] != labels[index] ) np++;
					}
				}
			}
			if( np > 1 )
			{
				contourx[cind] = k;
				contoury[cind] = j;
				istaken[mainindex] = true;
				//img[mainindex] = color;
				cind++;
			}
			mainindex++;
		}
	}

	imgOut = imgIn.clone();

	int numboundpix = cind;//int(contourx.size());
	for( int j = 0; j < numboundpix; j++ )
	{
		int ii = contoury[j]*width + contourx[j];
		// ubuff[ii] = 0xffffff;
		imgOut[contoury[j]][contourx[j]] = cv::Vec3b(255, 255, 255);

		for( int n = 0; n < 8; n++ )
		{
			int x = contourx[j] + dx8[n];
			int y = contoury[j] + dy8[n];
			if( (x >= 0 && x < width) && (y >= 0 && y < height) )
			{
				int ind = y*width + x;
				// if(!istaken[ind]) ubuff[ind] = 0;
				if (!istaken[ind]) imgOut[y][x] = cv::Vec3b(0, 0, 0);
			}
		}
	}
}

// M rows and N columns grids segments
// the labels are from 0 to M*N-1
int CreateGridSegments( const cv::Mat_<cv::Vec3b> &imgIn, cv::Mat_<int> &segLabels, int M, int N )
{
	int width = imgIn.cols;
	int height = imgIn.rows;
	int lr = height/M;
	int lc = width/N;

	//if (height % M != 0) ++lr;
	//if (width % N != 0) ++lc;

	int rectH = lr*M;
	int rectW = lc*N;
	segLabels.create(rectH, rectW);
	int iy, ix;
	for (iy=0; iy<rectH; ++iy)
	{
		for (ix=0; ix<rectW; ++ix)
		{
			segLabels[iy][ix] = iy/lr*N+ix/lc;
		}
	}

	cv::copyMakeBorder(segLabels, segLabels, 0, height-rectH, 0, width-rectW, cv::BORDER_REPLICATE);

	return M*N;
}


int CreateSLICSegments( const cv::Mat_<cv::Vec3b> &imgIn, cv::Mat_<int> &segLabels, int numSp, int spSize, int createType, double compactness /*= 20.0*/ )
{
	int width = imgIn.cols;
	int height = imgIn.rows;
	int sz = width*height;
	
	unsigned int* img = new unsigned int [sz];

	int iy, ix;
	unsigned int *uInd = img;
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			*uInd = ((unsigned int)imgIn[iy][ix][2] << 16)
				+ ((unsigned int)imgIn[iy][ix][1] << 8)
				+ ((unsigned int)imgIn[iy][ix][0]);
			++uInd;
		}
	}


	int m_spcount = numSp;
	double m_compactness = compactness;

	//---------------------------------------------------------
	if(m_spcount < 20 || m_spcount > sz/4) m_spcount = sz/200;//i.e the default size of the superpixel is 200 pixels
	if(m_compactness < 1.0 || m_compactness > 80.0) m_compactness = 20.0;
	//---------------------------------------------------------
	int* labels = new int[sz];
	int numlabels(0);
	SLIC slic;
	if (createType == 1) slic.DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(img, width, height, labels, numlabels, m_spcount, m_compactness);
	else slic.DoSuperpixelSegmentation_ForGivenSuperpixelSize(img,width,height,labels,numlabels, spSize,compactness); 
	
	//slic.DoSuperpixelSegmentation_ForGivenSuperpixelSize(img, width, height, labels, numlabels, 10, m_compactness);//demo
	//slic.DrawContoursAroundSegments(img, labels, width, height, 0);

	segLabels.create(height, width);
	int *iInd = labels;
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			segLabels[iy][ix] = *iInd;
			++iInd;
		}
	}


	return numlabels;
}

void GetSubImageRangeFromSegments( const cv::Mat_<int> &segLabels, int numOfLabels, int kerLen, cv::Mat_<cv::Vec4i> &subRange, cv::Mat_<cv::Vec4i> &spRange )
{
	int width = segLabels.cols;
	int height = segLabels.rows;

	// 0 -- 3
	// left, up, right, down
	cv::Mat_<cv::Vec4i> labelExtreme(numOfLabels, 1);
	
	// initialization
	int iy, ix;
	for (iy=0; iy<numOfLabels; ++iy)
	{
		labelExtreme[iy][0][0] = width;
		labelExtreme[iy][0][2] = -1;
		labelExtreme[iy][0][1] = height;
		labelExtreme[iy][0][3] = -1;
	}

	// find extremes of superpixel
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			int tmp = segLabels[iy][ix];
			(ix < labelExtreme[tmp][0][0])? labelExtreme[tmp][0][0] = ix: NULL;
			(ix > labelExtreme[tmp][0][2])? labelExtreme[tmp][0][2] = ix: NULL;
			(iy < labelExtreme[tmp][0][1])? labelExtreme[tmp][0][1] = iy: NULL;
			(iy > labelExtreme[tmp][0][3])? labelExtreme[tmp][0][3] = iy: NULL;
		}
	}

	cv::Mat_<cv::Vec4i> subimageExtreme(numOfLabels, 1);
	// add kernel size to the range
	for (iy=0; iy<numOfLabels; ++iy)
	{
		subimageExtreme[iy][0][0] = max<int>(labelExtreme[iy][0][0]-kerLen, 0);
		subimageExtreme[iy][0][2] = min<int>(labelExtreme[iy][0][2]+kerLen, width-1);
		subimageExtreme[iy][0][1] = max<int>(labelExtreme[iy][0][1]-kerLen, 0);
		subimageExtreme[iy][0][3] = min<int>(labelExtreme[iy][0][3]+kerLen, height-1);
	}


	subRange.create(height, width);
	spRange.create(height, width);
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			// 0 -- 3
			// left, up, right, down
			int tmpLabel = segLabels[iy][ix];
			subRange[iy][ix] = subimageExtreme[tmpLabel][0];
			spRange[iy][ix] = labelExtreme[tmpLabel][0];
		}
	}
}

void GetSubImageRangeFromSegments( const cv::Mat_<int> &segLabels, int numOfLabels, int kerLen, vector<cv::Vec4i> &subRange, vector<cv::Vec4i> &spRange )
{
	int width = segLabels.cols;
	int height = segLabels.rows;

	// 0 -- 3
	// left, up, right, down
	cv::Mat_<cv::Vec4i> labelExtreme(numOfLabels, 1);
	
	// initialization
	int iy, ix;
	for (iy=0; iy<numOfLabels; ++iy)
	{
		labelExtreme[iy][0][0] = width;
		labelExtreme[iy][0][2] = -1;
		labelExtreme[iy][0][1] = height;
		labelExtreme[iy][0][3] = -1;
	}

	// find extremes of superpixel
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			int tmp = segLabels[iy][ix];
			(ix < labelExtreme[tmp][0][0])? labelExtreme[tmp][0][0] = ix: NULL;
			(ix > labelExtreme[tmp][0][2])? labelExtreme[tmp][0][2] = ix: NULL;
			(iy < labelExtreme[tmp][0][1])? labelExtreme[tmp][0][1] = iy: NULL;
			(iy > labelExtreme[tmp][0][3])? labelExtreme[tmp][0][3] = iy: NULL;
		}
	}

	cv::Mat_<cv::Vec4i> subimageExtreme(numOfLabels, 1);
	// add kernel size to the range
	for (iy=0; iy<numOfLabels; ++iy)
	{
		subimageExtreme[iy][0][0] = max<int>(labelExtreme[iy][0][0]-kerLen, 0);
		subimageExtreme[iy][0][2] = min<int>(labelExtreme[iy][0][2]+kerLen, width-1);
		subimageExtreme[iy][0][1] = max<int>(labelExtreme[iy][0][1]-kerLen, 0);
		subimageExtreme[iy][0][3] = min<int>(labelExtreme[iy][0][3]+kerLen, height-1);
	}


	subRange.resize(height*width);
	spRange.resize(height*width);
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			// 0 -- 3
			// left, up, right, down
			int tmpLabel = segLabels[iy][ix];
			subRange[iy*width+ix] = subimageExtreme[tmpLabel][0];
			spRange[iy*width+ix] = labelExtreme[tmpLabel][0];
		}
	}
}

void GetPCentricSubImageRange( const cv::Mat_<cv::Vec3b> &imgIn, int spLen, int kerLen, cv::Mat_<cv::Vec4i> &subRange, cv::Mat_<cv::Vec4i> &spRange )
{
	int width = imgIn.cols;
	int height = imgIn.rows;

	subRange.create(height, width);
	int iy, ix;
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			// 0 -- 3
			// left, up, right, down
			subRange[iy][ix][0] = max<int>(ix-spLen, 0);
			subRange[iy][ix][2] = min<int>(ix+spLen, width-1);
			subRange[iy][ix][1] = max<int>(iy-spLen, 0);
			subRange[iy][ix][3] = min<int>(iy+spLen, height-1);
		}
	}

	spRange.create(height, width);
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			// 0 -- 3
			// left, up, right, down
			spRange[iy][ix][0] = max<int>(subRange[iy][ix][0]-kerLen, 0);
			spRange[iy][ix][2] = min<int>(subRange[iy][ix][2]+kerLen, width-1);
			spRange[iy][ix][1] = max<int>(subRange[iy][ix][1]-kerLen, 0);
			spRange[iy][ix][3] = min<int>(subRange[iy][ix][3]+kerLen, height-1);
		}
	}
}

void GetSegmentsRepresentativePixels(const cv::Mat_<int> &segLabels, int numOfLabels, cv::Mat_<cv::Vec2i> &rePixel)
{
	int iy, ix, height, width;
	height = segLabels.rows;
	width = segLabels.cols;

	rePixel.create(numOfLabels, 1);
	cv::Mat_<int> visitedLabel(numOfLabels, 1);
	visitedLabel.setTo(cv::Scalar(0.0));
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			int tmpLabel = segLabels[iy][ix];
			if (!visitedLabel[tmpLabel][0])
			{
				visitedLabel[tmpLabel][0] = 1;
				rePixel[tmpLabel][0] = cv::Vec2i(iy, ix);
			}
		}
	}
}

void GetSegmentsRepresentativePixelsRandomAssign(const cv::Mat_<int> &segLabels, int numOfLabels, cv::Mat_<cv::Vec2i> &rePixel)
{
	int iy, ix, height, width;
	height = segLabels.rows;
	width = segLabels.cols;

	rePixel.create(numOfLabels, 1);
	vector<vector<cv::Vec2i>> spPixels;
	spPixels.resize(numOfLabels);
	for (iy=0; iy<numOfLabels; ++iy)
		spPixels[iy].clear();
	for (iy=0; iy<height; ++iy)
	{
		for (ix=0; ix<width; ++ix)
		{
			int tmpLabel = segLabels[iy][ix];
			spPixels[tmpLabel].push_back(cv::Vec2i(iy, ix));
		}
	}

	cv::RNG rng;
	for (iy=0; iy<numOfLabels; ++iy)
	{
		rePixel[iy][0] = spPixels[iy][rng.next() % spPixels[iy].size()];
	}
}