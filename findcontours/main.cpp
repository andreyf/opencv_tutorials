#include <cv.h>
#include <highgui.h>

using namespace cv;

int main() {
    // Read the image (EN)
    // Читаем изображение (RU)
    Mat img0 = imread("apple.jpg", 1);

    Mat img1;

    // RGB to GRAY (EN)
    // Преобразуем RGB изображение в оттенки серого (RU)
    cvtColor(img0, img1, CV_RGB2GRAY);

    // Apply Canny filter (EN)
    // Применяем фильтр Canny (RU)
    Canny(img1, img1, 100, 200);

    // Find contours (EN)
    // Находим контуры (RU)
    vector< vector<Point> > contours;
    findContours(img1, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    // Create mask (EN)
    // Создаем маску (RU)
    Mat mask = Mat::zeros(img1.rows, img1.cols, CV_8UC1);

    // CV_FILLED fills the connected components found (EN)
    // CV_FILLED заполняет найденные компоненты (RU)
    drawContours(mask, contours, -1, Scalar(255), CV_FILLED);

    // Create a new image (EN)
    // Создаем еще одно изображение (RU)
    Mat crop(img0.rows, img0.cols, CV_8UC3);

    // Set background to green (EN)
    // Делаем задний фон зеленым (RU)
    crop.setTo(Scalar(0,255,0));

    // Create the result image (EN)
    // Создаем результирующее изображение (RU)
    img0.copyTo(crop, mask);

    // Normalize the mask image (EN)
    // Нормализуем маску (RU)
    normalize(mask.clone(), mask, 0.0, 255.0, CV_MINMAX, CV_8UC1);

    // Show images (EN)
    // Показываем изображения (RU)
    imshow("original", img0);
    imshow("mask", mask);
    imshow("canny", img1);
    imshow("cropped", crop);

    // Save images (EN)
    // Сохраняем изображения (RU)
    imwrite("apple_canny.jpg", img1);
    imwrite("apple_mask.jpg", mask);
    imwrite("apple_cropped.jpg", crop);

    waitKey(0);

    return 0;
}