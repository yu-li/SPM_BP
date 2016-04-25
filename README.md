Sped-up PatchMatch Belief Propagation (SPM-BP)
========================

This is an implementation of SPM-BP for *Optical Flow* estimation that correspondes to our published paper:

Y. Li, D. Min, M. S. Brown, M. N. Do, J. Lu. **"SPM-BP: Sped-up PatchMatch Belief Propagation for Continuous MRFs"**. in *ICCV 2015*. 

Project Website: [Efficient Inference for Continuous MRFs]
(https://publish.illinois.edu/visual-modeling-and-analytics/efficient-inference-for-continuous-mrfs/)

## Usage
- The whole codes are in the `code` folder. You can use CMake to compile SPM-BP (Tested only on 64 bit Ubuntu 14.04 and Windows 7 with Visual Studio 2012).
- For windows user, a compiled execuable with demo usage is provided in `Release` folder.
- We will be happy if you cite us when using this code!
- If you want to test *Stereo Matching* using SPM-BP, we can share the execuable upon request.

## Dependency
- [OpenCV 3.0](http://opencv.org/opencv-3-0.html)
- [SLIC superpixel [1]](http://ivrg.epfl.ch/research/superpixels) (included)
- [Cross-based Local Multipoint Filtering (CLMF) [2]]
(https://sites.google.com/site/filteringtutorial/) (included)
- [Fast Global Image Smoothing (FGS) [3]](https://sites.google.com/site/globalsmoothing/) (modified and included) 

## References
[1] R. Achanta , A. Shaji, K. Smith, A. Lucchi,P. Fua, and S. Susstrunk, " SLIC superpixels compared to state-of-the-art superpixel methods,"  IEEE Trans. on Pattern Analysis and Machine Intelligence (TPAMI), 34(11), 2274-2282, 2012.

[2] J. Lu, K. Shi, D. Min, L. Lin, and M. N. Do, "Cross-based local multipoint filtering," in Proc. IEEE Int. Conf. Computer Vision and Pattern Recognition (CVPR), Providence, Rhode Island, 2012.

[3] D. Min, S. Choi, J. Lu, B. Ham, K. Sohn, and M. N. Do, “Fast Global Image Smoothing Based on Weighted Least Squares,” IEEE Trans. on Image Processing (TIP), 23(12), 5638-5653, 2014.

## License
Copyright (c) 2015, [Yu Li](http://yu-li.github.io/) All rights reserved.

For research and education purpose only. 

