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
	VideoCapture cap(0);

    //checks whether the capture is initialized or not
    if(!cap.isOpened()) {
        return -1;
    }
    
	for(;;) {
        Mat frame;
        cap >> frame; //Take a pic
        imwrite("FaceFiles/yourFace.png", frame);
        if(waitKey(30) >= 0) break; //Loops continuously until process is terminated
    }
	return 0;
}
