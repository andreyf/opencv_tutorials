#include <iostream>
#include <highgui.h>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

int main() {

    Mat src = imread("coins.jpg");
    if (src.data == nullptr) return -1;

    imshow("src", src);

    // Create binary image from source image (EN)
    // Создаем бинарное изображение из исходного (RU)
    Mat bw;
    cvtColor(src, bw, CV_BGR2GRAY);
    threshold(bw, bw, 40, 255, CV_THRESH_BINARY);
    imshow("bw", bw);

    // Perform the distance transform algorithm (EN)
    // Выполняем алгоритм distance transform (RU)
    Mat dist;
    distanceTransform(bw, dist, CV_DIST_L2, 3);

    // Normalize the distance image for range = {0.0, 1.0} (EN)
    // Нормализуем изображение в диапозоне {0.0 1.0} (RU)
    normalize(dist, dist, 0, 1., NORM_MINMAX);
    imshow("dist", dist);

    // Threshold to obtain the peaks
    // This will be the markers for the foreground objects (EN)
    // Выполняем Threshold для определения пиков
    // Это будут маркеры для объектов на переднем плане (RU)
    threshold(dist, dist, .5, 1., CV_THRESH_BINARY);
    imshow("dist2", dist);

    // Create the CV_8U version of the distance image
    // It is needed for cv::findContours() (EN)
    // Создаем CV_8U версию distance изображения
    // Это нужно для фуекции cv::findContours() (RU)
    Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);

    // Find total markers (EN)
    // Находим все маркеры (RU)
    vector<vector<Point> > contours;
    findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    auto ncomp = static_cast<int>(contours.size());

    // Create the marker image for the watershed algorithm (EN)
    // Создаем маркерное изображение для алгоритма watershed (RU)
    Mat markers = Mat::zeros(dist.size(), CV_32SC1);

    // Draw the foreground markers (EN)
    // Рисуем маркеры переднего плана (RU)
    for (int i =0; i < ncomp; i++)
        drawContours(markers, contours, i, Scalar::all(i+1), -1);


    // Draw background marker
    circle(markers, Point(5,5), 3, CV_RGB(255,255,255), -1);
    imshow("markers", markers*10000);

    // Perform the watershed algorithm (EN)
    // Выполняем алгоритм watershed (RU)
    watershed(src, markers);

    // Generate random colors (EN)
    // Генерируем случайные цвета (RU)
    vector<Vec3b> colors;

    for (int i = 0; i < ncomp; i++)
    {
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);

        colors.emplace_back(static_cast<uchar>(b), static_cast<uchar>(g), static_cast<uchar>(r));
    }

    // Create the result image (EN)
    // Создаем результирующее изображение (RU)
    Mat dst = Mat::zeros(markers.size(), CV_8UC3);

    // Fill labeled objects with random colors (EN)
    // Заполняем помеченные объекты случайным цветом (RU)
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i,j);
            dst.at<Vec3b>(i,j) = index > 0 && index <= ncomp ? colors[index-1] : Vec3b(0,0,0);
        }
    }

    imshow("dst", dst);

    waitKey(0);

    return 0;
}