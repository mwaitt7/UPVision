#UPVision README

##About UPVision
UPVision is a system designed for detecting forklift operator fatigue. UPVision is the product of a capstone project at University of Portland involving four Computer Science and Electrical Engineering students, and is sponsored by Hyster-Yale Materials Handling.

##Project Dependencies
- To build this project, we have been using CMake, which is a makefile generator available for free at https://cmake.org/download/. You may be able to use different methods for making the project, but CMake has been our go-to.
- You will also need to install dlib and OpenCV, which are the image processing libraries we chose for the project. These are available at: http://dlib.net/ and https://opencv.org/releases.html.
- While it is certainly not a requirement, the instructions for building this project assume that the user is using a Linux-based machine. It is possible to build on Windows, but we have limited experience with that.

##How to Build/Run
To build the project on a Linux-based machine, follow the following steps: 

1. Download the .zip file and navigate to the project directory via the terminal.
2. If there is a build folder in the project directory, move "shape\_predictor\_68\_face\_landmarks.dat" and "CMakeLists.txt" to the project directory, and delete the build folder.
3. Next, run the following commands in succession.
	- mkdir build
	- mv shape\_predictor\_68\_face\_landmarks.dat /build
	- mv CMakeLists.txt /build
	- cd /build
	- cmake .
	- cmake --build . --config Release
	- ./imageCapture
4. If you make any changes, run "cmake --build . --config Release" in the /build directory to compile and rebuild.
