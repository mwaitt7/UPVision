# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.11

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.11.0/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.11.0/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/michael/Dropbox/Señor/Capstone/UPVision/dlib-19.9/dlib/cmake_utils/test_for_cpp11

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/michael/Dropbox/Señor/Capstone/UPVision/build/dlib_build/cpp11_test_build

# Include any dependencies generated for this target.
include CMakeFiles/cpp11_test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/cpp11_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cpp11_test.dir/flags.make

CMakeFiles/cpp11_test.dir/cpp11_test.cpp.o: CMakeFiles/cpp11_test.dir/flags.make
CMakeFiles/cpp11_test.dir/cpp11_test.cpp.o: /Users/michael/Dropbox/Señor/Capstone/UPVision/dlib-19.9/dlib/cmake_utils/test_for_cpp11/cpp11_test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --progress-dir=/Users/michael/Dropbox/Señor/Capstone/UPVision/build/dlib_build/cpp11_test_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cpp11_test.dir/cpp11_test.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cpp11_test.dir/cpp11_test.cpp.o -c /Users/michael/Dropbox/Señor/Capstone/UPVision/dlib-19.9/dlib/cmake_utils/test_for_cpp11/cpp11_test.cpp

CMakeFiles/cpp11_test.dir/cpp11_test.cpp.i: cmake_force
	@echo "Preprocessing CXX source to CMakeFiles/cpp11_test.dir/cpp11_test.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/michael/Dropbox/Señor/Capstone/UPVision/dlib-19.9/dlib/cmake_utils/test_for_cpp11/cpp11_test.cpp > CMakeFiles/cpp11_test.dir/cpp11_test.cpp.i

CMakeFiles/cpp11_test.dir/cpp11_test.cpp.s: cmake_force
	@echo "Compiling CXX source to assembly CMakeFiles/cpp11_test.dir/cpp11_test.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/michael/Dropbox/Señor/Capstone/UPVision/dlib-19.9/dlib/cmake_utils/test_for_cpp11/cpp11_test.cpp -o CMakeFiles/cpp11_test.dir/cpp11_test.cpp.s

# Object files for target cpp11_test
cpp11_test_OBJECTS = \
"CMakeFiles/cpp11_test.dir/cpp11_test.cpp.o"

# External object files for target cpp11_test
cpp11_test_EXTERNAL_OBJECTS =

libcpp11_test.a: CMakeFiles/cpp11_test.dir/cpp11_test.cpp.o
libcpp11_test.a: CMakeFiles/cpp11_test.dir/build.make
libcpp11_test.a: CMakeFiles/cpp11_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --progress-dir=/Users/michael/Dropbox/Señor/Capstone/UPVision/build/dlib_build/cpp11_test_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libcpp11_test.a"
	$(CMAKE_COMMAND) -P CMakeFiles/cpp11_test.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cpp11_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cpp11_test.dir/build: libcpp11_test.a

.PHONY : CMakeFiles/cpp11_test.dir/build

CMakeFiles/cpp11_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cpp11_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cpp11_test.dir/clean

CMakeFiles/cpp11_test.dir/depend:
	cd /Users/michael/Dropbox/Señor/Capstone/UPVision/build/dlib_build/cpp11_test_build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/michael/Dropbox/Señor/Capstone/UPVision/dlib-19.9/dlib/cmake_utils/test_for_cpp11 /Users/michael/Dropbox/Señor/Capstone/UPVision/dlib-19.9/dlib/cmake_utils/test_for_cpp11 /Users/michael/Dropbox/Señor/Capstone/UPVision/build/dlib_build/cpp11_test_build /Users/michael/Dropbox/Señor/Capstone/UPVision/build/dlib_build/cpp11_test_build /Users/michael/Dropbox/Señor/Capstone/UPVision/build/dlib_build/cpp11_test_build/CMakeFiles/cpp11_test.dir/DependInfo.cmake
.PHONY : CMakeFiles/cpp11_test.dir/depend

