//
// Created by andrey on 13.07.16.
//

#include "ght.h"

using namespace cv;

void runGHT(char c) {
    //c = 't'; // create template
    //c = 'r'; // run algorithm
    if (c == 't') {
        GHT ght;
        ght.createTemplate();
    }
    else if (c == 'r') {
        GHT ght;
        //ght.setTresholds(180, 250);
        ght.createRtable();
        //Mat detect_img = imread("files\\Img_01.png", 1);
        Mat detect_img = imread("files/Img_03.png", 1);
        //Mat detect_img = imread("files\\Img_03.png", 1);
        ght.accumulate(detect_img);
        ght.bestCandidate();
    }
}

int main (int argc, char *argv[])
{
    runGHT('r');
}

