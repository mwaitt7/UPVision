
//
//  imageCapture.cpp
//
//
//  Created by the UPVision team on 1/24/18.
//
#include <utility>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

using namespace dlib;
using namespace std;
std::vector<full_object_detection> facialFeatures;
int main()
{
	try
	{

		cv::VideoCapture cap(0);
		//checks whether the capture is initialized or not
		if (!cap.isOpened())
		{
			cerr << "Unable to connect to camera" << endl;
			return 1;
		}
		
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor pose_model;
		image_window awindow;
		deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
		while (!awindow.is_closed())
		{
			cv::Mat frame;
			if (!cap.read(frame))
			{
				break;
			}
			
			cv_image<bgr_pixel> cimg(frame);
			
			std::vector<rectangle> faces = detector(cimg);
			facialFeatures.clear();
			for (unsigned long i = 0; i < faces.size(); ++i){
				facialFeatures.push_back(pose_model(cimg, faces[i]));
			
				}
			
			awindow.clear_overlay();
			awindow.set_image(cimg);
			awindow.add_overlay(render_face_detections(facialFeatures));
		}
	}
	catch (serialization_error& e)
	{
		cout << "Cannot find .dat file check the location" << endl;
		cout << endl << e.what() << endl;
	}
	catch (exception& e)
	{
		cout << e.what() << endl;
	}
}

