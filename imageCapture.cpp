//
//  imageCapture.cpp
//
//
//  Created by the UPVision team on 1/24/18.
//

#include "opencv2/opencv.hpp"
#include <iostream>
#include <unistd.h>

using namespace std;
using namespace cv;

int main() {
	VideoCapture cap(0); //Take a pic

	for(;;) {
        Mat frame;
        cap >> frame; // get a new frame from camera
        imwrite("FaceFiles/yourFace.png", frame);
        if(waitKey(30) >= 0) break;   // you can increase delay to 2 seconds here
    }
	return 0;
}