//
//  imageCapture.cpp
//
//	Author: UP Vision Team
//  	Date: 1/24/18
//	Version: Alpha 7
//	Description: The goal of this class it calculate the drowsiness using a camera.
//
#define _USE_MATH_DEFINES

#include <math.h>
#include <utility>
#include <cmath>
#include "dlib-19.9/dlib/opencv.h"
#include <opencv2/highgui/highgui.hpp>
#include "dlib-19.9/dlib/image_processing/frontal_face_detector.h"
#include "dlib-19.9/dlib/image_processing/render_face_detections.h"
#include "dlib-19.9/dlib/image_processing.h"
#include "dlib-19.9/dlib/gui_widgets.h"
#include <opencv2/calib3d/calib3d.hpp>

#define USE_RP_CAM 1

#if USE_RP_CAM
#include "raspicam/raspicam_cv.h"
#endif

using namespace dlib;
using namespace std;

const float HEAD_ORIENTATION_CONFIDENCE_WEIGTH = 0.20;
const float EYE_CLOSED_CONFIDENCE_WEIGHT = 0.18;
const float ACTIVITY_LEVEL_WEIGHT = 0.2;

std::vector<full_object_detection> facialFeatures; //Contains the landmark values.
float confidence_Level = 0.0;
int UI_R = 0;
int UI_G = 255;
int UI_B = 0;
bool distanceInitialized = false;
int desiredConfMethod = 1; //Testing
double offsetFromBase = 0;
float tempDistance = 0;
int PRINT_TO_FILE = 0;

/*
Method name:get_camera_matrix
Description: Approximate the cameras internal specs.
Input Paramater:the focal length of the camera. And the center point of the 2D picture.
Return Paramater: matrix of the camera internal specs
*/
cv::Mat get_camera_matrix(float focal_length, cv::Point2d center_point)
{
	cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, center_point.x, 0, focal_length, center_point.y, 0, 0, 1);
	return camera_matrix;
}

/*
Method name:get the blinking duration
Description: This method calculates how long are the eyes closed by looking at the corner of the eyes and the upper and lower eye lids. 
Return Paramater:eye_closed_dur (How long have the eyes been closed).
*/
float get_eyeclosed_duration(std::vector<full_object_detection> facialFeatures, bool& firstTime, std::clock_t &begin, double &starttime2, double &timeElapsed, bool &eye_Is_Closed,int & counter) {
	//Used for eye blinking
	float eye_closed_dur = 0.0f;
	double EAR = 0;
	double EYE_THRESHOLD = 0.25; //indicates a blink when EAR is below this value
	double EYE_CONSECUTIVE_FRAMES = 2; //the # of consecutive frames the eye must be below threshold

	if (facialFeatures.size() < 1) {
		return 0;
	}
	double P37_41_x = facialFeatures[0].part(37).x() - facialFeatures[0].part(41).x();
	double P37_41_y = facialFeatures[0].part(37).y() - facialFeatures[0].part(41).y();

	double P38_40_x = facialFeatures[0].part(38).x() - facialFeatures[0].part(40).x();
	double P38_40_y = facialFeatures[0].part(38).y() - facialFeatures[0].part(40).y();

	// compute the distances between the two sets of vertical eye landmarks
	double Dist_37_41 = sqrt((P37_41_x*P37_41_x) + (P37_41_y*P37_41_y));
	double Dist_38_40 = sqrt((P38_40_x*P38_40_x) + (P38_40_y*P38_40_y));

	double P36_39_x = facialFeatures[0].part(36).x() - facialFeatures[0].part(39).x();
	double P36_39_y = facialFeatures[0].part(36).y() - facialFeatures[0].part(39).y();

	// compute the distances between the horizontal eye landmark
	double Dist_36_39 = sqrt((P36_39_x*P36_39_x) + (P36_39_y*P36_39_y));


	// compute the eye aspect ratio for the right eye
	double EAR_right = (Dist_37_41 + Dist_38_40) / (2.0 * Dist_36_39);

	double P43_47_x = facialFeatures[0].part(43).x() - facialFeatures[0].part(47).x();
	double P43_47_y = facialFeatures[0].part(43).y() - facialFeatures[0].part(47).y();

	double P44_46_x = facialFeatures[0].part(44).x() - facialFeatures[0].part(46).x();
	double P44_46_y = facialFeatures[0].part(44).y() - facialFeatures[0].part(46).y();

	// compute the distances between the two sets of vertical eye landmarks
	double Dist_43_47 = sqrt((P43_47_x*P43_47_x) + (P43_47_y*P43_47_y));
	double Dist_44_46 = sqrt((P44_46_x*P44_46_x) + (P44_46_y*P44_46_y));

	double P42_45_x = facialFeatures[0].part(42).x() - facialFeatures[0].part(45).x();
	double P42_45_y = facialFeatures[0].part(42).y() - facialFeatures[0].part(45).y();

	// compute the distances between the horizontal eye landmark
	double Dist_42_45 = sqrt((P42_45_x*P42_45_x) + (P42_45_y*P42_45_y));

	// compute the eye aspect ratio for the left eye
	double EAR_left = (Dist_43_47 + Dist_44_46) / (2.0 * Dist_42_45);


	// average the EAR for both left and right eye
	EAR = (EAR_left + EAR_right) / 2.0;
	//if the avg eye aspect ratio is less than the threshold, increase frame counter
	if (EAR < EYE_THRESHOLD) {
		if (firstTime) {
			begin = std::clock();
			starttime2 = (std::clock() - begin) / (double)CLOCKS_PER_SEC;
			firstTime = false;
		}
		timeElapsed = (std::clock() - begin) / (double)CLOCKS_PER_SEC;
		counter++;
		eye_Is_Closed = true;

	}
	else {

		//if the frame counter is greater than or equal to the allowed number of frames
		//below EYE_THRESHOLD, increase the number of blinks.
		//if (counter >= EYE_CONSECUTIVE_FRAMES) {
		//	blink_dur = timeElapsed - starttime2;
		//totalBlinks++;		
		//}
		eye_Is_Closed = false;
		starttime2 = timeElapsed;
		counter = 0; //reset counter  
	}
	
	if (eye_Is_Closed) {
	eye_closed_dur = timeElapsed - starttime2;
	eye_closed_dur = eye_closed_dur * 2;
	}
	else {

	eye_closed_dur = 0;
	}
	return eye_closed_dur;

}


/*
Method name:get_offset_from_base
Description:Calculate the head orientation and output the offset from base. There are three methods for determining the offset the first method converts a 2D image into a 3D model and then calculate the euler rotation angle.
			The second method calculate the distance from two points in the nose in order to assess the head orientation. The third method relies on the second method but this time with a normalized distance based on the area
			of the surroanding box.
Return Paramater: the offsetfrombase which represent a distance that is used to calculate the confidence level.
*/
double get_offset_from_base(std::vector<rectangle> faces, std::vector<cv::Point2d> &image_pts, std::vector<cv::Point3d> &object_pts, cv::Mat camera_matrix, cv::Mat &dist_coeffs, cv::Mat &rotation_matrix, cv::Mat &rotation_vector,
							cv::Mat &translation_vector, cv::Mat &position_matrix, cv::Mat &out_intrinsics, cv::Mat &out_rotation,cv::Mat &out_translation, cv::Mat &euler_angle) {
	if (desiredConfMethod == 1) {
		if (facialFeatures.size() != 0) {

			//Fill the coordinates from facial detection
			image_pts.push_back(cv::Point2d(facialFeatures[0].part(17).x(), facialFeatures[0].part(17).y())); // 17 is the left corner of the left brow
			image_pts.push_back(cv::Point2d(facialFeatures[0].part(21).x(), facialFeatures[0].part(21).y())); // 21 is the right corner of the left brow
			image_pts.push_back(cv::Point2d(facialFeatures[0].part(22).x(), facialFeatures[0].part(22).y())); // 22 is the left corner of the right brow
			image_pts.push_back(cv::Point2d(facialFeatures[0].part(26).x(), facialFeatures[0].part(26).y())); // 26 is the right corner of the right brow
			image_pts.push_back(cv::Point2d(facialFeatures[0].part(36).x(), facialFeatures[0].part(36).y())); // 36 is the left corner of the left eye
			image_pts.push_back(cv::Point2d(facialFeatures[0].part(39).x(), facialFeatures[0].part(39).y())); // 39 is the right corner of the left eye
			image_pts.push_back(cv::Point2d(facialFeatures[0].part(42).x(), facialFeatures[0].part(42).y())); // 42 is the left corner of the right eye
			image_pts.push_back(cv::Point2d(facialFeatures[0].part(45).x(), facialFeatures[0].part(45).y())); // 45 is the right corner of the right eye
			image_pts.push_back(cv::Point2d(facialFeatures[0].part(31).x(), facialFeatures[0].part(31).y())); // 31 is the left corner of the nose
			image_pts.push_back(cv::Point2d(facialFeatures[0].part(35).x(), facialFeatures[0].part(35).y())); // 35 is the right corner of the nose
			image_pts.push_back(cv::Point2d(facialFeatures[0].part(48).x(), facialFeatures[0].part(48).y())); // 48 is the left corner of the mouth
			image_pts.push_back(cv::Point2d(facialFeatures[0].part(54).x(), facialFeatures[0].part(54).y())); // 54 is the right corner of the mouth
			image_pts.push_back(cv::Point2d(facialFeatures[0].part(57).x(), facialFeatures[0].part(57).y())); // 57 is bottom center of the mouth
			image_pts.push_back(cv::Point2d(facialFeatures[0].part(8).x(), facialFeatures[0].part(8).y()));   // 8 is the tip of the chin
																											  //Calculate the position of the 3D model.
			cv::solvePnP(object_pts, image_pts, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

			//Calculate euler's angle (basically the XYZ)
			cv::Rodrigues(rotation_vector, rotation_matrix); //Converts rotation vector to a matrix
			cv::hconcat(rotation_matrix, translation_vector, position_matrix); // horizontal concatination
			cv::decomposeProjectionMatrix(position_matrix, out_intrinsics, out_rotation, out_translation, cv::noArray(), cv::noArray(), cv::noArray(), euler_angle);
		}
		offsetFromBase = 0;
		if (euler_angle.at<double>(0) > 10) {
			offsetFromBase = euler_angle.at<double>(0) - 10;
		}

		

	}



	if (desiredConfMethod == 2) {
		/*
		CONFIDENCE LEVEL TESTING - METHOD #2: NOSE DISTANCE
		Sets a baseline for distance between point 31-34 in y-direction,
		then checks the offset from the current distance from the baseline.
		TODO: can make this better by doing same calculation with distance
		from point 20->38, and 25->45.
		If that offset is negative => looking down, sleepy
		If positive => looking up, alert
		*/
		double baseDistance = 0;
		offsetFromBase = 0;
		if (facialFeatures.size() != 0) {
			//Get baseline
			if (!distanceInitialized) {
				baseDistance = facialFeatures[0].part(34)(1) - facialFeatures[0].part(31)(1);
				distanceInitialized = true;
			}
			else {
				//Is this distance shorter than the baseline? (This means nodding off)
				double tempDistance = facialFeatures[0].part(34)(1) - facialFeatures[0].part(31)(1);
				offsetFromBase = tempDistance - baseDistance;

				if (offsetFromBase <= 0) {
					//Flip the sign for use in calculations
					offsetFromBase = 0 - offsetFromBase;
				}
			}
		}
	}


	if (desiredConfMethod == 3) {
		/*
		Confidence level calculation - Method #1: Nose Y-Pos
		*/
		float base_Y_Pos = 0.0f;
		tempDistance = 0;
		float ori_Distance = 0;
		//If there's a face in frame, check angle
		if (facialFeatures.size() != 0) {
			base_Y_Pos = (facialFeatures[0].part(8)(1) + facialFeatures[0].part(22)(1)) / 2;

			float current_Y_Pos = (facialFeatures[0].part(57)(1) - facialFeatures[0].part(33)(1));
			tempDistance = current_Y_Pos / faces[0].area();
			if (tempDistance >0.0035) {
				//place holders for how fast do we want the confidence level to grow.
				ori_Distance = 10;
				offsetFromBase = 10;
			}
		}
	}
	return offsetFromBase;
}



int main() {
	try {
		//Initialize the video capture and set the camera to the first one
#if USE_RP_CAM
		raspicam::RaspiCam_Cv cap;
#else
		cv::VideoCapture cap(0);
#endif
		cv::Mat frame;
		
		//Readjust the resolution of the video camera to speed up processing time.
		cap.set(CV_CAP_PROP_FRAME_WIDTH, 480/2);
		cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240/2);

		// Michael changed this to isOpened() to fix compiler errors
#if USE_RP_CAM
		if (!cap.open()) {
#else
		if (!cap.isOpened()) {
#endif
			cerr << "Unable to open camera" << endl;
			return 1;
		}
		//Load up the face detection
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

		//==== Variable Declarations for the 2D to 3D Modeling START =================
		//fill in 3D ref points(world coordinates), model referenced from http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
		std::vector<cv::Point3d> object_pts;
		object_pts.push_back(cv::Point3d(6.825897, 6.760612, 4.402142));     //#33 left brow left corner
		object_pts.push_back(cv::Point3d(1.330353, 7.122144, 6.903745));     //#29 left brow right corner
		object_pts.push_back(cv::Point3d(-1.330353, 7.122144, 6.903745));    //#34 right brow left corner
		object_pts.push_back(cv::Point3d(-6.825897, 6.760612, 4.402142));    //#38 right brow right corner
		object_pts.push_back(cv::Point3d(5.311432, 5.485328, 3.987654));     //#13 left eye left corner
		object_pts.push_back(cv::Point3d(1.789930, 5.393625, 4.413414));     //#17 left eye right corner
		object_pts.push_back(cv::Point3d(-1.789930, 5.393625, 4.413414));    //#25 right eye left corner
		object_pts.push_back(cv::Point3d(-5.311432, 5.485328, 3.987654));    //#21 right eye right corner
		object_pts.push_back(cv::Point3d(2.005628, 1.409845, 6.165652));     //#55 nose left corner
		object_pts.push_back(cv::Point3d(-2.005628, 1.409845, 6.165652));    //#49 nose right corner
		object_pts.push_back(cv::Point3d(2.774015, -2.080775, 5.048531));    //#43 mouth left corner
		object_pts.push_back(cv::Point3d(-2.774015, -2.080775, 5.048531));   //#39 mouth right corner
		object_pts.push_back(cv::Point3d(0.000000, -3.116408, 6.097667));    //#45 mouth central bottom corner
		object_pts.push_back(cv::Point3d(0.000000, -7.415691, 4.070434)); //#6 chin corner

		
		//Our 2D image points 
		std::vector<cv::Point2d> image_pts;
		//Rotations (to get x y and z)
		cv::Mat rotation_vector;
		cv::Mat rotation_matrix;
		cv::Mat translation_vector;
		cv::Mat position_matrix = cv::Mat(3, 4, CV_64FC1);    
		cv::Mat euler_angle = cv::Mat(3, 1, CV_64FC1);

		//temp buffer for decomposeProjectionMatrix()
		cv::Mat out_intrinsics = cv::Mat(3, 3, CV_64FC1);
		cv::Mat out_rotation = cv::Mat(3, 3, CV_64FC1);
		cv::Mat out_translation = cv::Mat(3, 1, CV_64FC1);
		//===== Variable Declarations for the 2D to 3D Modeling END ===================================

		//====== Eye closing variables start=======
		int counter = 0;//frame counter
		int totalBlinks = 0;
		double starttime2 = 0;
		bool firstTime = true;
		float blink_dur = 0.0f;
		std::clock_t begin;
		double timeElapsed;
		bool eye_Is_Closed = false;
		//=========The eye closing variables END==========

		//=========Face detection Improvment Algorithm START =======
		std::vector<rectangle> temp;
		int change_In_X = 0;
		int change_In_Y = 0;
		bool flag1 = true;
		int personIsStill = 0;

		//=========Face detection Improvment Algorithm END =======
		while (!awindow.is_closed()) {
			//uncomment to use for raspberry pi camera
#if USE_RP_CAM
			cap.grab();
			cap.retrieve(frame);
#endif			


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
				fps = (double)frames / (duration - starttime);
				starttime = duration;
				frames = 0;
			}

			//Face Detection
#if !USE_RP_CAM			
			if (!cap.read(frame)) {
				break;
			}
#endif

			// Camera matrix calculations 
			double focal_length = frame.cols; 
			cv::Point2d center_point = cv::Point2d(frame.cols / 2, frame.rows / 2);
			cv::Mat camera_matrix = get_camera_matrix(focal_length,center_point);
			cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type); // Assuming no lens distortion
			cv_image<bgr_pixel> cimg(frame);
			//Assume no distortion
			cv::Mat distortion = cv::Mat::zeros(4, 1, cv::DataType<double>::type);



			std::clock_t faceDetectStart = std::clock();
			std::vector<rectangle> faces = detector(cimg);
			double faceDetectTime = std::clock()-faceDetectStart;
			if (flag1) {
				temp = faces;
				flag1 = false;
			}
			//Attempt to approximate where the face is when we lose track of it.
			if (faces.size() == 0) {
				faces = temp;
				
			}
			if (faces.size() > 0) {
				change_In_X = temp[0].left() - faces[0].left();
				change_In_Y = temp[0].top() - faces[0].top();

				//Activity calculation
				if ((abs(change_In_X) < 20) && abs(change_In_Y) < 20) {
					personIsStill = 1;
				}
				else {
					personIsStill = 0;
				}
			}
			facialFeatures.clear();
			for (unsigned long i = 0; i < faces.size(); ++i) {
				facialFeatures.push_back(pose_model(cimg, faces[i]));

			}
			//Find out how long the person's eyes have been closed.
			blink_dur = get_eyeclosed_duration(facialFeatures, firstTime, begin, starttime2, timeElapsed, eye_Is_Closed, counter);
			//Find the offsetFromBase
			std::clock_t getOffsetStart = std::clock();
			offsetFromBase = get_offset_from_base(faces, image_pts, object_pts, camera_matrix, dist_coeffs, rotation_matrix, rotation_vector, translation_vector, position_matrix, out_intrinsics, out_rotation, out_translation, euler_angle);
			double getOffsetTime = std::clock()-getOffsetStart;			

			//Confidence level calculation
			confidence_Level += (offsetFromBase*HEAD_ORIENTATION_CONFIDENCE_WEIGTH - HEAD_ORIENTATION_CONFIDENCE_WEIGTH) + (EYE_CLOSED_CONFIDENCE_WEIGHT*blink_dur - EYE_CLOSED_CONFIDENCE_WEIGHT) + (ACTIVITY_LEVEL_WEIGHT*personIsStill);


			//Make sure it doesn't exceed the bounds
			if (confidence_Level < 0) {
				confidence_Level = 0;
			}
			if (confidence_Level > 100) {
				confidence_Level = 100;
			}

			//Set color for UI overlay
			if (confidence_Level < 50) {
				UI_R = 0;
				UI_G = 255;
			}
			else if (confidence_Level < 70) {
				UI_R = 255;
				UI_G = 255;
			}
			else {
				UI_R = 255;
				UI_G = 0;
			}

			//reset the window and redraw everything.
			std::clock_t renderStart = std::clock();
			awindow.clear_overlay();
			awindow.set_image(cimg);
			awindow.add_overlay(render_face_detections(facialFeatures));
			rectangle rect;
			rectangle rem;
			temp = faces;
			if (desiredConfMethod == 2) {
				awindow.add_overlay(image_window::overlay_rect(rect, rgb_pixel(UI_R, UI_G, UI_B), "FRAMES PER SECOND: " + std::to_string(fps) +  "\nBLINK DURATION IN SECONDS: " + std::to_string(blink_dur) + "\nDISTANCE OF Y FROM BASE :" + std::to_string(offsetFromBase) + "\nConfidence Level :" + std::to_string(confidence_Level)));
			}
			else if (desiredConfMethod == 3) {
				awindow.add_overlay(image_window::overlay_rect(rect, rgb_pixel(UI_R, UI_G, UI_B), "FRAMES PER SECOND: " + std::to_string(fps) +  "\nBLINK DURATION IN SECONDS: " + std::to_string(blink_dur) + "\nDISTANCE OF Y FROM BASE :" + std::to_string(tempDistance) + "\nConfidence Level :" + std::to_string(confidence_Level)));
				if (faces.size() > 0) {
					awindow.add_overlay(faces[0]);
				}
			}
			else if (desiredConfMethod == 1) {
				awindow.add_overlay(image_window::overlay_rect(rect, rgb_pixel(UI_R, UI_G, UI_B), "FRAMES PER SECOND: " + std::to_string(fps) + "\nBLINK DURATION IN SECONDS: " + std::to_string(blink_dur) + "\nX :" + std::to_string(euler_angle.at<double>(0)) + "\nY :" + std::to_string(euler_angle.at<double>(1))+ "\nZ :" + std::to_string(euler_angle.at<double>(2))  + "\nConfidence Level :" + std::to_string(confidence_Level) + "\nIs Person Still? :" + std::to_string(personIsStill)));

				if (faces.size() > 0) {
					awindow.add_overlay(faces[0]);
				}
				image_pts.clear();
			}
			double renderTime = std::clock()-renderStart;

			
			cout << "faceDetectTime "<<faceDetectTime/ (double)CLOCKS_PER_SEC <<"s\trenderGUITime " <<renderTime/ (double)CLOCKS_PER_SEC<<"s\tgetOffsetTime "<<getOffsetTime/ (double)CLOCKS_PER_SEC<<"s\n\n"; 

			//Printing result to a file for debugging and training purposes.
			if (PRINT_TO_FILE > 0) {

				//writes the output to a file with a timestamp
				time_t now = time(0);
				tm *currTime = localtime(&now);
				int hour = currTime->tm_hour;
				int min = currTime->tm_min;
				int sec = currTime->tm_sec;
				ofstream file;

				file.open("output.txt", ofstream::out | ofstream::app);

				if (confidence_Level > 0) {
					file << "###############################################" << endl;
					file << to_string(hour) << ":" << to_string(min) << ":" << to_string(sec) << endl;
					file << "FPS: " << std::to_string(fps) << " frames/sec" << endl;
					file << "Blink Duartion: " << std::to_string(blink_dur) << " sec" << endl;
					file << "Y dist from base: " << std::to_string(offsetFromBase) << endl;
					file << "Confidence Level: " << std::to_string(confidence_Level) << endl;
					file << "###############################################" << endl;

				}


				file.close();
			}
		}
	}
	catch (serialization_error& e) {
		cout << "Cannot find .dat file, check the location." << endl;
		cout << endl << e.what() << endl;
	}
	catch (exception& e) {
		cout << "jhee" << endl;
		cout << e.what() << endl;
	}
}
