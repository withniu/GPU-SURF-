/*
 * Copyright (C) 2009-2010 Andre Schulz, Florian Jung, Sebastian Hartte,
 *						   Daniel Trick, Christan Wojek, Konrad Schindler,
 *						   Jens Ackermann, Michael Goesele
 * Copyright (C) 2008-2009 Christopher Evans <chris.evans@irisys.co.uk>, MSc University of Bristol
 *
 * This file is part of SURFGPU.
 *
 * SURFGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SURFGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SURFGPU.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "surflib.h"
#include "kmeans.h"
#include "utils.h"
#include <ctime>
#include <iostream>

//-------------------------------------------------------
//  - 5 to show matches between static images

//-------------------------------------------------------

int mainStaticMatch(void);


//-------------------------------------------------------

int main(void) 
{
	mainStaticMatch();
	return 0;
}


//-------------------------------------------------------

int mainStaticMatch()
{

  time_t start,end1,end2,end3,end4,end5;
  start = clock();

  IplImage *img1, *img2;
  img1 = cvLoadImage("../data/1.JPG");
  img2 = cvLoadImage("../data/2.JPG");
  

  end1 = clock();

  IpVec ipts1, ipts2;
  surfDetDes(img1,ipts1,false,4,4,2,0.0008f);
  surfDetDes(img2,ipts2,false,4,4,2,0.0008f);
  
  std::cout << "im1" << std::endl;
  std::cout << "Size:" << ipts1.size() << std::endl;

  std::cout << "im2" << std::endl;
  std::cout << "Size:" << ipts2.size() << std::endl;
  end2 = clock();
  
  IpPairVec matches;
  getMatches(ipts1,ipts2,matches);

  end3 = clock();
  
  for (unsigned int i = 0; i < matches.size(); ++i)
  {
    drawPoint(img1,matches[i].first);
    drawPoint(img2,matches[i].second);
  
    const int & w = img1->width;
    cvLine(img1,cvPoint(matches[i].first.x,matches[i].first.y),cvPoint(matches[i].second.x+w,matches[i].second.y), cvScalar(255,255,255),1);
    cvLine(img2,cvPoint(matches[i].first.x-w,matches[i].first.y),cvPoint(matches[i].second.x,matches[i].second.y), cvScalar(255,255,255),1);
  }

  std::cout << "Matches: " << matches.size() << std::endl;
/*
  cvNamedWindow("1", CV_WINDOW_AUTOSIZE );
  cvNamedWindow("2", CV_WINDOW_AUTOSIZE );
  cvShowImage("1", img1);
  cvShowImage("2", img2);
  cvWaitKey(0);
*/
	end4 = clock();

//  cvSaveImage("result_gpu1.jpg",img1);
//	cvSaveImage("result_gpu2.jpg",img2);

	// Stitch two images
	IplImage *img = cvCreateImage(cvSize(img1->width + img2->width,
										 img1->height),img1->depth,img1->nChannels); 
	cvSetImageROI( img, cvRect( 0, 0, img1->width, img1->height ) ); 
    cvCopy(img1, img);
    cvSetImageROI( img, cvRect(img1->width,0, img2->width, img2->height) ); 
    cvCopy(img2, img); 
	cvResetImageROI(img); 	
	cvSaveImage("result_gpu.jpg",img);
	
	end5 = clock();
	double dif1 = (double)(end1 - start) / CLOCKS_PER_SEC;
	double dif2 = (double)(end2 - end1) / CLOCKS_PER_SEC;
	double dif3 = (double)(end3 - end2) / CLOCKS_PER_SEC;
	double dif4 = (double)(end4 - end3) / CLOCKS_PER_SEC;
	double dif5 = (double)(end5 - end4) / CLOCKS_PER_SEC;
	double total = (double)(end5 - start) / CLOCKS_PER_SEC;
	std::cout.setf(std::ios::fixed,std::ios::floatfield);
	std::cout.precision(5);
	std::cout << "Time(load):" << dif1 << std::endl;
	std::cout << "Time(descriptor):" << dif2 << std::endl;
 	std::cout << "Time(match):" << dif3 << std::endl;
 	std::cout << "Time(plot):" << dif4 << std::endl;
 	std::cout << "Time(save):" << dif5 << std::endl;
  	std::cout << "Time(Total):" << total << std::endl;
  return 0;
}
