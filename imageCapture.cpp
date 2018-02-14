
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
		cap.set(CV_CAP_PROP_FRAME_WIDTH,480);
		cap.set(CV_CAP_PROP_FRAME_HEIGHT,240);
		
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

        //Used for eye blinking
        double EYE_THRESHOLD = 0.25; //indicates a blink when EAR is below this value
        double EYE_CONSECUTIVE_FRAMES = 2; //the # of consecutive frames the eye must be below threshold
        double EAR = 0;
        int counter = 0;//frame counter
        int totalBlinks = 0;
        
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
                
                double P37_41_x = facialFeatures[i].part(37).x() - facialFeatures[i].part(41).x();
                double P37_41_y = facialFeatures[i].part(37).y() - facialFeatures[i].part(41).y();
                
                double P38_40_x = facialFeatures[i].part(38).x() - facialFeatures[i].part(40).x();
                double P38_40_y = facialFeatures[i].part(38).y() - facialFeatures[i].part(40).y();
                
                // compute the distances between the two sets of vertical eye landmarks
                double Dist_37_41 = sqrt((P37_41_x*P37_41_x)+ (P37_41_y*P37_41_y));
                double Dist_38_40 = sqrt((P38_40_x*P38_40_x)+ (P38_40_y*P38_40_y));
                
                double P36_39_x = facialFeatures[i].part(36).x() - facialFeatures[i].part(39).x();
                double P36_39_y = facialFeatures[i].part(36).y() - facialFeatures[i].part(39).y();
                
                // compute the distances between the horizontal eye landmark
                double Dist_36_39 = sqrt((P36_39_x*P36_39_x)+ (P36_39_y*P36_39_y));
                
                
                // compute the eye aspect ratio for the right eye
                double EAR_right = (Dist_37_41 + Dist_38_40) / (2.0 * Dist_36_39);
                
                double P43_47_x = facialFeatures[i].part(43).x() - facialFeatures[i].part(47).x();
                double P43_47_y = facialFeatures[i].part(43).y() - facialFeatures[i].part(47).y();
                
                double P44_46_x = facialFeatures[i].part(44).x() - facialFeatures[i].part(46).x();
                double P44_46_y = facialFeatures[i].part(44).y() - facialFeatures[i].part(46).y();
                
                // compute the distances between the two sets of vertical eye landmarks
                double Dist_43_47 = sqrt((P43_47_x*P43_47_x)+ (P43_47_y*P43_47_y));
                double Dist_44_46 = sqrt((P44_46_x*P44_46_x)+ (P44_46_y*P44_46_y));
                
                double P42_45_x = facialFeatures[i].part(42).x() - facialFeatures[i].part(45).x();
                double P42_45_y = facialFeatures[i].part(42).y() - facialFeatures[i].part(45).y();
                
                // compute the distances between the horizontal eye landmark
                double Dist_42_45 = sqrt((P42_45_x*P42_45_x)+ (P42_45_y*P42_45_y));
                
                
                // compute the eye aspect ratio for the left eye
                double EAR_left = (Dist_43_47 + Dist_44_46) / (2.0 * Dist_42_45);
                
                
                // average the EAR for both left and right eye
                EAR =  (EAR_left + EAR_right) / 2.0;
                

                //if the avg eye aspect ratio is less than the threshold, increase frame counter
                if (EAR < EYE_THRESHOLD) {
                    counter += 1;
                }
                else {
                    //if the frame counter is greater than or equal to the allowed number of frames
                    //below EYE_THRESHOLD, increase the number of blinks.
                    if (counter >= EYE_CONSECUTIVE_FRAMES) {
                        totalBlinks += 1;
                    }
                    
                    counter = 0; //reset counter
                }
                
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
            awindow.add_overlay(image_window::overlay_rect(rect, rgb_pixel(0, 0, 255), "UPVision v0.1\n\nHEAD ORIENTATION: "+ std::to_string(headAngle) + " degrees\nFRAMES PER SECOND: "+ std::to_string(fps)+ "\nEYE ASPECT RATIO: " + std::to_string(EAR) + "\nBLINK COUNT: " + std::to_string(totalBlinks)));
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
