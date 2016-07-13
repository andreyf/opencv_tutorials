//
// Created by andrey on 13.07.16.
//

#ifndef PROJECT_GHT_H
#define PROJECT_GHT_H

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

const float pi = 3.14159265f;

struct Rpoint{
	int dx;
	int dy;
	float phi;
};

struct Rpoint2{
	float x;
	float y;
	int phiindex;
};

class GHT{

private:
	// accumulator matrix
	cv::Mat accum;

	// accumulator matrix
	cv::Mat showimage;

	// contour points:
	std::vector<Rpoint> pts;

	// reference point (inside contour(
	cv::Vec2i refPoint;

	// R-table of template object:
	std::vector<std::vector<cv::Vec2i> > Rtable;

	// number of intervals for angles of R-table:
	int intervals;

	// threasholds of canny edge detector
	int thr1;
	int thr2;

	// width of template contour
	int wtemplate;

	// minimum and maximum width of scaled contour
	int wmin;
	int wmax;

	// minimum and maximum rotation allowed for template
	float phimin;
	float phimax;

	// dimension in pixels of squares in image
	int rangeXY;

	// interval to increase scale
	int rangeS;

public:

	GHT(){
		// default values

		// canny threasholds
		thr1 = 50;
		thr2 = 150;

		// minimun and maximum width of the searched template
		wmin = 50;
		wmax = 200;

		// increasing step in pixels of the width
		rangeS = 5;

		// side of the squares in which the image is divided
		rangeXY = 6;

		// min value allowed is -pi
		phimin = -pi;

		// max value allowed is +pi
		phimax = +pi;

		// number of slices (angles) in R-table
		intervals = 16;
	}

//	void setTresholds(int t1, int t2);

//	void setLinearPars(int w1, int w2, int rS, int rXY);

//	void setAngularPars(int p1, int p2, int ints);

	void createTemplate();

	void createRtable();

	void accumulate(cv::Mat& input_img);

	void bestCandidate();

//	void localMaxima();


private:

	void readPoints();

	void readRtable();

	int inline roundToInt(float num);

//    short inline at4D(cv::Mat &mt, int i0, int i1, int i2, int i3);

    short inline * pointer4D(cv::Mat &mt, int i0, int i1, int i2, int i3);

};

#endif //PROJECT_GHT_H
