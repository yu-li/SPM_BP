SPM-BP - ICCV 2015
========================

This is an implementation of the SPM-BP Optical Flow Estimator that correspondes to our paper published in ICCV 2015 

"SPM-BP: Sped-up PatchMatch Belief Propagation for Continuous MRFs". Y. Li, D. Min, M. S. Brown, M. N. Do, J. Lu

Project Website: 
[https://publish.illinois.edu/visual-modeling-and-analytics/efficient-inference-for-continuous-mrfs/]

## Usage
- The whole codes are in the `code` folder. You can use CMake to compile SPM-BP (Tested only in Windows 7 with Visual Studio 2012; but the code should be able to run in Linux or Mac with slight modification)
- For windows user, a compiled execuable with demo usage is provided in `Release` folder.

## Dependency
- OpenCV 3.0 [http://opencv.org/opencv-3-0.html]
- SLIC superpixel, (included), [http://ivrg.epfl.ch/research/superpixels](http://ivrg.epfl.ch/research/superpixels]
- Fast Global Image Smoothing (FGS), (modified and included) [https://sites.google.com/site/globalsmoothing/]

## License
Copyright (c) 2014, Yu Li All rights reserved.

For research and education purpose only. 
