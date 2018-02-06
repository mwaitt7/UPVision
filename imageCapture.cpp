
//
//  imageCapture.cpp
//
//
//  Created by the UPVision team on 1/24/18.
//
#define _USE_MATH_DEFINES

#include <math.h>
#include <utility>
#include "dlib-19.9/dlib/opencv.h"
#include <opencv2/highgui/highgui.hpp>
#include "dlib-19.9/dlib/image_processing/frontal_face_detector.h"
#include "dlib-19.9/dlib/image_processing/render_face_detections.h"
#include "dlib-19.9/dlib/image_processing.h"
#include "dlib-19.9/dlib/gui_widgets.h"

using namespace dlib;
using namespace std;

std::vector<full_object_detection> facialFeatures;
int main() {
	try {

		cv::VideoCapture cap(0);
		//checks whether the capture is initialized or not
		if (!cap.isOpened()) {
			cerr << "Unable to connect to camera" << endl;
			return 1;
		}
		
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor pose_model;
		image_window awindow;
		deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
		
		//Used for FPS calculation
		int frames = 0;
		double starttime = 0;
		bool first = true;
		float fps = 0.0f;
		std::clock_t start;
    	double duration;

		while (!awindow.is_closed()) {
			//FPS Calculation
			if (first) {
				start = std::clock();
				frames = 0;
				starttime = (std::clock() - start) / (double)CLOCKS_PER_SEC;
				first = false;
			}
			duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
			frames++;

			if (frames > 10) {
			    fps = (double) frames / (duration - starttime);
			    starttime = duration;
			    frames = 0;
			}

			//Face Detection
			cv::Mat frame;
			if (!cap.read(frame)) {
				break;
			}
			
			cv_image<bgr_pixel> cimg(frame);
			
			std::vector<rectangle> faces = detector(cimg);
			facialFeatures.clear();
			for (unsigned long i = 0; i < faces.size(); ++i) {
				facialFeatures.push_back(pose_model(cimg, faces[i]));
			}

			float headAngle = 0;
			//If there's a face in frame, check angle
			if (facialFeatures.size() != 0) {
				int x1 = facialFeatures[0].part(0)(0);
				int x2 = facialFeatures[0].part(16)(0);

				int y1 = facialFeatures[0].part(0)(1);
				int y2 = facialFeatures[0].part(16)(1);

				headAngle = atan2(y1 - y2, x1 - x2);
				headAngle = headAngle*180/M_PI;
			}

			awindow.clear_overlay();
			awindow.set_image(cimg);
			awindow.add_overlay(render_face_detections(facialFeatures));
			rectangle rect;
			awindow.add_overlay(image_window::overlay_rect(rect, rgb_pixel(0, 0, 255), "UPVision v0.1\n\nHEAD ORIENTATION: "+ std::to_string(headAngle) + " degrees\nFRAMES PER SECOND: "+ std::to_string(fps)));
		}
	}
	catch (serialization_error& e) {
		cout << "Cannot find .dat file, check the location." << endl;
		cout << endl << e.what() << endl;
	}
	catch (exception& e) {
		cout << e.what() << endl;
	}
}

