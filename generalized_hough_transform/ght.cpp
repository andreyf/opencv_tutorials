#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "ght.h"

using namespace std;
using namespace cv;

short inline *GHT::pointer4D(Mat &mt, int i0, int i1, int i2, int i3) {
    return (short*)(mt.data + i0*mt.step.p[0] + i1*mt.step.p[1] + i2*mt.step.p[2] + i3*mt.step.p[3]);
}

/*short inline GHT::at4D(Mat &mt, int i0, int i1, int i2, int i3) {
    return *( (short*)(mt.data + i0*mt.step.p[0] + i1*mt.step.p[1] + i2*mt.step.p[2] + i3*mt.step.p[3]));
}*/

int inline GHT::roundToInt(float num) {
	return (num > 0.0) ? (int)(num + 0.5f) : (int)(num - 0.5f);
}

void GHT::readRtable() {
	Rtable.clear();
	Rtable.resize((unsigned long) intervals);

	// put points in the right interval, according to discretized angle and range size
	float range = pi/intervals;
	for (vector<Rpoint>::size_type t = 0; t < pts.size(); ++t){
		int angleindex = (int)((pts[t].phi+pi/2)/range);
		if (angleindex == intervals) angleindex=intervals-1;
		Rtable[angleindex].push_back( Vec2i(pts[t].dx, pts[t].dy) );
	}
}

void GHT::readPoints() {
	// read original template image and its worked-out contour
	Mat original_img = imread("files/template_original.jpg", 1);
	Mat input_img_gray;
	input_img_gray.create( Size(original_img.cols, original_img.rows), CV_8UC1);
	cvtColor(original_img, input_img_gray, CV_BGR2GRAY);
	//Mat template_img = imread("files\\contour_def.bmp", 1);
	Mat template_img = imread("files/contour_def2.bmp", 1);

	// find reference point inside contour image and save it in variable refPoint
	int nl= template_img.rows;
	int nc= template_img.cols;
	for (int j=0; j<nl; ++j) {
		Vec3b* data= (Vec3b*)(template_img.data + template_img.step.p[0]*j);
		for (int i=0; i<nc; ++i) {
			if ( data[i]==Vec3b(127,127,127)  ){
				refPoint = Vec2i(i,j);
			}
		}
	}

	// get Scharr matrices from original template image to obtain contour gradients
	Mat dx;
	dx.create( Size(original_img.cols, original_img.rows), CV_16SC1);
	Sobel(input_img_gray, dx, CV_16S, 1, 0, CV_SCHARR);
	Mat dy;
	dy.create( Size(original_img.cols, original_img.rows), CV_16SC1);
	Sobel(input_img_gray, dy, CV_16S, 0, 1, CV_SCHARR);

	// load points on vector
	pts.clear();
	int mindx = INT_MAX;
	int maxdx = INT_MIN;
	for (int j=0; j<nl; ++j) {
		Vec3b* data= (Vec3b*)(template_img.data + template_img.step.p[0]*j);
		for (int i=0; i<nc; ++i) {
			if ( data[i]==Vec3b(255,255,255)  )
			{
				short vx = dx.at<short>(j,i);
				short vy = dy.at<short>(j,i);
				Rpoint rpt;
				//float mag = std::sqrt( float(vx*vx+vy*vy) );
				rpt.dx = refPoint(0)-i;
				rpt.dy = refPoint(1)-j;
				float a = atan2((float)vy, (float)vx); //radians
				rpt.phi = ((a > 0) ? a-pi/2 : a+pi/2);
				//float a = atan2((float)vy, (float)vx) * 180/3.14159265358979f; //degrees
				//rpt.phi = ((a > 0) ? a-90 : a+90);
				// update further right and left dx
				if (rpt.dx < mindx) mindx=rpt.dx;
				if (rpt.dx > maxdx) maxdx=rpt.dx;
				pts.push_back( rpt );
			}
		}
	}
	// maximum width of the contour
	wtemplate = maxdx-mindx+1;
}

/*void GHT::localMaxima() {
	// to bve implemented ...
}*/

void GHT::bestCandidate() {
	double minval;
	double maxval;
	int id_min[4] = { 0, 0, 0, 0};
	int id_max[4] = { 0, 0, 0, 0};
	minMaxIdx(accum, &minval, &maxval, id_min, id_max);

	int nl= showimage.rows;
	int nc= showimage.cols;
	Mat	input_img2 = showimage.clone();

	Vec2i referenceP = Vec2i(id_max[0]*rangeXY+(rangeXY+1)/2, id_max[1]*rangeXY+(rangeXY+1)/2);

	// rotate and scale points all at once. Then impress them on image
    float deltaphi = pi/intervals;
	int r0 = static_cast<int>(-floor(phimin / deltaphi));
	int reff = id_max[3]-r0;
	float cs = cos(reff*deltaphi);
	float sn = sin(reff*deltaphi);
	int w = wmin + id_max[2]*rangeS;
	float wratio = (float)w/(wtemplate);
	for (std::vector<std::vector<Vec2i> >::size_type ii = 0; ii < Rtable.size(); ++ii){
		for (std::vector<Vec2i>::size_type jj= 0; jj < Rtable[ii].size(); ++jj){
//				int iimod = (ii+reff) % intervals;
			int dx = roundToInt(wratio*(cs*Rtable[ii][jj][0] - sn*Rtable[ii][jj][1]));
			int dy = roundToInt(wratio*(sn*Rtable[ii][jj][0] + cs*Rtable[ii][jj][1]));
			int x = referenceP[0] - dx;
			int y = referenceP[1] - dy;
			//Rtablerotatedscaled[ii].push_back(Vec2i( dx, dy));
			if ( (x<nc)&&(y<nl)&&(x>-1)&&(y>-1) ){
				input_img2.at<Vec3b>(y, x) = Vec3b(0, 255, 255);
			}
		}
	}
	// show result
	bool alt = false;
	for(;;)
	{
		char c = (char)waitKey(750);
		if( c == 27 )
			break;
		if (alt){
			imshow("input_img", input_img2);
		}
		else{
			imshow("input_img", showimage);
		}
		alt = !alt;
	}
}

void GHT::accumulate(cv::Mat& input_img) {
	showimage = input_img;

	// transform image to grayscale:
	Mat src_gray;
	src_gray.create( Size(input_img.cols, input_img.rows), CV_8UC1);
	cvtColor(input_img, src_gray, CV_BGR2GRAY);

	// reduce noise with a kernel 3x3 and get cannyedge image:
	Mat detected_edges;
	blur( src_gray, detected_edges, Size(3,3) );
	Canny( detected_edges, detected_edges, thr1, thr2, 3 );
	//imshow("detected_edges", detected_edges);

	// get Scharr matrices from image to obtain contour gradients
	Mat dx;
	dx.create( Size(input_img.cols, input_img.rows), CV_16SC1);
	Sobel(src_gray, dx, CV_16S, 1, 0, CV_SCHARR);
	Mat dy;
	dy.create( Size(input_img.cols, input_img.rows), CV_16SC1);
	Sobel(src_gray, dy, CV_16S, 0, 1, CV_SCHARR);

	// load all points from image all image contours on vector pts2
	int nl= detected_edges.rows;
	int nc= detected_edges.cols;
	float deltaphi = pi/intervals;
	float inv_deltaphi = (float)intervals/pi;
	float inv_rangeXY = (float)1/rangeXY;
	float pi_half = pi*0.5f;
	std::vector<Rpoint2> pts2;
	for (int j=0; j<nl; ++j) {
		uchar* data= detected_edges.data + detected_edges.step.p[0] * j;
		for (int i=0; i<nc; ++i) {
			if ( data[i]==255  ) // consider only white points (contour)
			{
				short vx = dx.at<short>(j,i);
				short vy = dy.at<short>(j,i);
				Rpoint2 rpt;
				rpt.x = i*inv_rangeXY;
				rpt.y = j*inv_rangeXY;
				float a = atan2((float)vy, (float)vx);              //	gradient angle in radians
				float phi = ((a > 0) ? a-pi_half : a+pi_half);      // contour angle with respect to x axis
				int angleindex = (int)((phi+pi*0.5f)*inv_deltaphi); // index associated with angle (0 index = -90 degrees)
				if (angleindex == intervals) angleindex=intervals-1;// -90�angle and +90� has same effect
				rpt.phiindex = angleindex;
				pts2.push_back( rpt );
			}
		}
	}

	// OpenCv 4-dimensional matrix definition and in general a useful way for defining multidimensional arrays and vectors in c++
	// create accumulator matrix
	int X = static_cast<int>(ceil((float)nc / rangeXY));
	int Y = static_cast<int>(ceil((float)nl / rangeXY));
	int S = static_cast<int>(ceil((float)(wmax - wmin) / rangeS + 1.0f));
	int R = static_cast<int>(ceil(phimax / deltaphi) - floor(phimin / deltaphi));
	if (phimax==pi && phimin==-pi) R--;
	int r0 = static_cast<int>(-floor(phimin / deltaphi));
	int matSizep_S[] = {X, Y, S, R};
	accum.create(4, matSizep_S, CV_16S);
	accum = Scalar::all(0);

	// icrease accum cells with hits corresponding with slope in Rtable vector rotatated and scaled
	float inv_wtemplate_rangeXY = (float)1/(wtemplate*rangeXY);

	// rotate RTable from minimum to maximum angle
	for (int r=0; r<R; ++r) {  // rotation
		int reff = r-r0;
		std::vector<std::vector<Vec2f> > Rtablerotated((unsigned long) intervals);

		// cos and sin are computed in the outer loop to reach computational efficiency
		float cs = std::cos(reff*deltaphi);
		float sn = std::sin(reff*deltaphi);
		for (std::vector<std::vector<Vec2i> >::size_type ii = 0; ii < Rtable.size(); ++ii){
			for (std::vector<Vec2i>::size_type jj= 0; jj < Rtable[ii].size(); ++jj){
				int iimod = (int) ((ii + reff) % intervals);
				Rtablerotated[iimod].push_back(Vec2f(cs*Rtable[ii][jj][0] - sn*Rtable[ii][jj][1], sn*Rtable[ii][jj][0] + cs*Rtable[ii][jj][1]));
			}
		}

		// scale the rotated RTable from minimum to maximum scale
		for (int s=0; s<S; ++s) {  // scale
			std::vector<std::vector<Vec2f> > Rtablescaled((unsigned long) intervals);
			int w = wmin + s*rangeS;
			float wratio = (float)w*inv_wtemplate_rangeXY;
			for (std::vector<std::vector<Vec2f> >::size_type ii = 0; ii < Rtablerotated.size(); ++ii){
				for (std::vector<Vec2f>::size_type jj= 0; jj < Rtablerotated[ii].size(); ++jj){
					Rtablescaled[ii].push_back(Vec2f(wratio*Rtablerotated[ii][jj][0], wratio*Rtablerotated[ii][jj][1]));
				}
			}

			// iterate through each point of edges and hit corresponding cells from rotated and scaled Rtable
			for (vector<Rpoint2>::size_type t = 0; t < pts2.size(); ++t){ // XY plane
				int angleindex = pts2[t].phiindex;
				for (std::vector<Vec2f>::size_type index = 0; index < Rtablescaled[angleindex].size(); ++index){
					float deltax = Rtablescaled[angleindex][index][0];
					float deltay = Rtablescaled[angleindex][index][1];
					int xcell = (int)(pts2[t].x + deltax);
					int ycell = (int)(pts2[t].y + deltay);
					if ( (xcell<X)&&(ycell<Y)&&(xcell>-1)&&(ycell>-1) ){
						//(*( (short*)(accum.data + xcell*accum.step.p[0] + ycell*accum.step.p[1] + s*accum.step.p[2]+ r*accum.step.p[3])))++;
						(*pointer4D(accum, xcell, ycell, s, r))++;
					}
				}
			}
		}
	}
}

void GHT::createRtable() {
	// code can be improved reading a pre-saved Rtable
	readPoints();
	readRtable();
}

void GHT::createTemplate() {
	Mat input_img = imread("files/template_original.jpg", 1);
	Mat src_gray;
	Mat detected_edges;
	src_gray.create( Size(input_img.cols, input_img.rows), CV_8UC1);
	cvtColor(input_img, src_gray, CV_BGR2GRAY);
	blur( src_gray, detected_edges, Size(3,3) );
	Canny( detected_edges, detected_edges, 1, 100, 3 );
	imwrite("files/contour_rough.bmp", detected_edges);
}

/*void GHT::setAngularPars(int p1, int p2, int ints) {
	if (p1<p2){
		if (p1>-pi){
			phimin = p1;
		}
		if (p2<+pi){
			phimax = p2;
		}
	}
	intervals = ints;
}*/

/*void GHT::setLinearPars(int w1, int w2, int rS, int rXY) {
	wmin = w1;
	wmax = w2;
	rangeS = rS;
	rangeXY = rXY;
}*/

/*void GHT::setTresholds(int t1, int t2) {
	thr1 = t1;
	thr2 = t2;
}*/

