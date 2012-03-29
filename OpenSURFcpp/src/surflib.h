/*********************************************************** 
*  --- OpenSURF ---                                       *
*  This library is distributed under the GNU GPL. Please   *
*  use the contact form at http://www.chrisevansdev.com    *
*  for more information.                                   *
*                                                          *
*  C. Evans, Research Into Robust Visual Features,         *
*  MSc University of Bristol, 2008.                        *
*                                                          *
************************************************************/

#ifndef SURFLIB_H
#define SURFLIB_H

#include <cv.h>
#include <highgui.h>

#include "integral.h"
#include "fasthessian.h"
#include "surf.h"
#include "ipoint.h"
#include "utils.h"


//! Library function builds vector of described interest points
inline void surfDetDes(IplImage *img,  /* image to find Ipoints in */
                       std::vector<Ipoint> &ipts, /* reference to vector of Ipoints */
                       bool upright = false, /* run in rotation invariant mode? */
                       int octaves = OCTAVES, /* number of octaves to calculate */
                       int intervals = INTERVALS, /* number of intervals per octave */
                       int init_sample = INIT_SAMPLE, /* initial sampling step */
                       float thres = THRES /* blob response threshold */)
{
  time_t start,end1,end2,end3,end4,end5;
  start = clock();
  
  // Create integral-image representation of the image
  IplImage *int_img = Integral(img);
end1 = clock();  
  // Create Fast Hessian Object
  FastHessian fh(int_img, ipts, octaves, intervals, init_sample, thres);
end2 = clock(); 
  // Extract interest points and store in vector ipts
  fh.getIpoints();
end3 = clock();  
  // Create Surf Descriptor Object
  Surf des(int_img, ipts);
  // Extract the descriptors for the ipts
  des.getDescriptors(upright);
end4 = clock();
  // Deallocate the integral image
  cvReleaseImage(&int_img);
end5 = clock();
	double dif1 = (double)(end1 - start) / CLOCKS_PER_SEC;
	double dif2 = (double)(end2 - end1) / CLOCKS_PER_SEC;
	double dif3 = (double)(end3 - end2) / CLOCKS_PER_SEC;
	double dif4 = (double)(end4 - end3) / CLOCKS_PER_SEC;
	double dif5 = (double)(end5 - end4) / CLOCKS_PER_SEC;
	std::cout.setf(std::ios::fixed,std::ios::floatfield);
	std::cout.precision(5);
	std::cout << "\tTime(Integral):" << dif1 << std::endl;
	std::cout << "\tTime(FastHessian):" << dif2 << std::endl;
 	std::cout << "\tTime(getIpoints):" << dif3 << std::endl;
 	std::cout << "\tTime(descriptor):" << dif4 << std::endl;
 	std::cout << "\tTime(cvReleaseImage):" << dif5 << std::endl;
}


//! Library function builds vector of interest points
inline void surfDet(IplImage *img,  /* image to find Ipoints in */
                    std::vector<Ipoint> &ipts, /* reference to vector of Ipoints */
                    int octaves = OCTAVES, /* number of octaves to calculate */
                    int intervals = INTERVALS, /* number of intervals per octave */
                    int init_sample = INIT_SAMPLE, /* initial sampling step */
                    float thres = THRES /* blob response threshold */)
{
  // Create integral image representation of the image
  IplImage *int_img = Integral(img);

  // Create Fast Hessian Object
  FastHessian fh(int_img, ipts, octaves, intervals, init_sample, thres);

  // Extract interest points and store in vector ipts
  fh.getIpoints();

  // Deallocate the integral image
  cvReleaseImage(&int_img);
}




//! Library function describes interest points in vector
inline void surfDes(IplImage *img,  /* image to find Ipoints in */
                    std::vector<Ipoint> &ipts, /* reference to vector of Ipoints */
                    bool upright = false) /* run in rotation invariant mode? */
{ 
  // Create integral image representation of the image
  IplImage *int_img = Integral(img);

  // Create Surf Descriptor Object
  Surf des(int_img, ipts);

  // Extract the descriptors for the ipts
  des.getDescriptors(upright);
  
  // Deallocate the integral image
  cvReleaseImage(&int_img);
}


#endif