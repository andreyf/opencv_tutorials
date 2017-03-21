#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main( )
{

    Mat image;
    image = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);

    if(! image.data )
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    // Создаем matrix, куда будем помещать обработанное изображение
    Mat gray;

    // Конвертироем RGB изображение в оттенки серого
    cvtColor(image, gray, CV_BGR2GRAY);

    namedWindow( "Display window", CV_WINDOW_AUTOSIZE );
    imshow( "Display window", image );

    namedWindow( "Result window", CV_WINDOW_AUTOSIZE );
    imshow( "Result window", gray );

    waitKey(0);
    return 0;
}