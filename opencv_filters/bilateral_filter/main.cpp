#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>

using namespace cv;

int main(int argc, char** argv)
{
   namedWindow("Before" , CV_WINDOW_AUTOSIZE);

   // Load the source image
   Mat src = imread( "../data/zebra.jpg", 1);

   // Create a destination Mat object
   Mat dst;

   // display the source image
   imshow("Before", src);

   for (int i=1; i<51; i=i+2)
   {
      // Bilateral smoothing
      bilateralFilter( src, dst, i, i*2, i*2 );

      //show the blurred image with the text
      imshow( "Bilateral filter", dst );

      //wait for 5 seconds
      waitKey(5000);
   }
}
