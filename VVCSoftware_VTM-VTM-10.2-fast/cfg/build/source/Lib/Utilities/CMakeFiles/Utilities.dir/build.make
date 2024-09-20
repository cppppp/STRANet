# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /S3/anaconda/lib/python3.10/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /S3/anaconda/lib/python3.10/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /S3/sty/STRANet_folder/VVCSoftware_VTM-VTM-10.2-fast

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /S3/sty/STRANet_folder/VVCSoftware_VTM-VTM-10.2-fast/build

# Include any dependencies generated for this target.
include source/Lib/Utilities/CMakeFiles/Utilities.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include source/Lib/Utilities/CMakeFiles/Utilities.dir/compiler_depend.make

# Include the progress variables for this target.
include source/Lib/Utilities/CMakeFiles/Utilities.dir/progress.make

# Include the compile flags for this target's objects.
include source/Lib/Utilities/CMakeFiles/Utilities.dir/flags.make

source/Lib/Utilities/CMakeFiles/Utilities.dir/VideoIOYuv.cpp.o: source/Lib/Utilities/CMakeFiles/Utilities.dir/flags.make
source/Lib/Utilities/CMakeFiles/Utilities.dir/VideoIOYuv.cpp.o: /S3/sty/STRANet_folder/VVCSoftware_VTM-VTM-10.2-fast/source/Lib/Utilities/VideoIOYuv.cpp
source/Lib/Utilities/CMakeFiles/Utilities.dir/VideoIOYuv.cpp.o: source/Lib/Utilities/CMakeFiles/Utilities.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/S3/sty/STRANet_folder/VVCSoftware_VTM-VTM-10.2-fast/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object source/Lib/Utilities/CMakeFiles/Utilities.dir/VideoIOYuv.cpp.o"
	cd /S3/sty/STRANet_folder/VVCSoftware_VTM-VTM-10.2-fast/build/source/Lib/Utilities && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT source/Lib/Utilities/CMakeFiles/Utilities.dir/VideoIOYuv.cpp.o -MF CMakeFiles/Utilities.dir/VideoIOYuv.cpp.o.d -o CMakeFiles/Utilities.dir/VideoIOYuv.cpp.o -c /S3/sty/STRANet_folder/VVCSoftware_VTM-VTM-10.2-fast/source/Lib/Utilities/VideoIOYuv.cpp

source/Lib/Utilities/CMakeFiles/Utilities.dir/VideoIOYuv.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/Utilities.dir/VideoIOYuv.cpp.i"
	cd /S3/sty/STRANet_folder/VVCSoftware_VTM-VTM-10.2-fast/build/source/Lib/Utilities && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /S3/sty/STRANet_folder/VVCSoftware_VTM-VTM-10.2-fast/source/Lib/Utilities/VideoIOYuv.cpp > CMakeFiles/Utilities.dir/VideoIOYuv.cpp.i

source/Lib/Utilities/CMakeFiles/Utilities.dir/VideoIOYuv.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/Utilities.dir/VideoIOYuv.cpp.s"
	cd /S3/sty/STRANet_folder/VVCSoftware_VTM-VTM-10.2-fast/build/source/Lib/Utilities && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /S3/sty/STRANet_folder/VVCSoftware_VTM-VTM-10.2-fast/source/Lib/Utilities/VideoIOYuv.cpp -o CMakeFiles/Utilities.dir/VideoIOYuv.cpp.s

source/Lib/Utilities/CMakeFiles/Utilities.dir/program_options_lite.cpp.o: source/Lib/Utilities/CMakeFiles/Utilities.dir/flags.make
source/Lib/Utilities/CMakeFiles/Utilities.dir/program_options_lite.cpp.o: /S3/sty/STRANet_folder/VVCSoftware_VTM-VTM-10.2-fast/source/Lib/Utilities/program_options_lite.cpp
source/Lib/Utilities/CMakeFiles/Utilities.dir/program_options_lite.cpp.o: source/Lib/Utilities/CMakeFiles/Utilities.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/S3/sty/STRANet_folder/VVCSoftware_VTM-VTM-10.2-fast/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object source/Lib/Utilities/CMakeFiles/Utilities.dir/program_options_lite.cpp.o"
	cd /S3/sty/STRANet_folder/VVCSoftware_VTM-VTM-10.2-fast/build/source/Lib/Utilities && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT source/Lib/Utilities/CMakeFiles/Utilities.dir/program_options_lite.cpp.o -MF CMakeFiles/Utilities.dir/program_options_lite.cpp.o.d -o CMakeFiles/Utilities.dir/program_options_lite.cpp.o -c /S3/sty/STRANet_folder/VVCSoftware_VTM-VTM-10.2-fast/source/Lib/Utilities/program_options_lite.cpp

source/Lib/Utilities/CMakeFiles/Utilities.dir/program_options_lite.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/Utilities.dir/program_options_lite.cpp.i"
	cd /S3/sty/STRANet_folder/VVCSoftware_VTM-VTM-10.2-fast/build/source/Lib/Utilities && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /S3/sty/STRANet_folder/VVCSoftware_VTM-VTM-10.2-fast/source/Lib/Utilities/program_options_lite.cpp > CMakeFiles/Utilities.dir/program_options_lite.cpp.i

source/Lib/Utilities/CMakeFiles/Utilities.dir/program_options_lite.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/Utilities.dir/program_options_lite.cpp.s"
	cd /S3/sty/STRANet_folder/VVCSoftware_VTM-VTM-10.2-fast/build/source/Lib/Utilities && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /S3/sty/STRANet_folder/VVCSoftware_VTM-VTM-10.2-fast/source/Lib/Utilities/program_options_lite.cpp -o CMakeFiles/Utilities.dir/program_options_lite.cpp.s

# Object files for target Utilities
Utilities_OBJECTS = \
"CMakeFiles/Utilities.dir/VideoIOYuv.cpp.o" \
"CMakeFiles/Utilities.dir/program_options_lite.cpp.o"

# External object files for target Utilities
Utilities_EXTERNAL_OBJECTS =

/S3/sty/STRANet_folder/VVCSoftware_VTM-VTM-10.2-fast/lib/umake/gcc-7.5/x86_64/release/libUtilities.a: source/Lib/Utilities/CMakeFiles/Utilities.dir/VideoIOYuv.cpp.o
/S3/sty/STRANet_folder/VVCSoftware_VTM-VTM-10.2-fast/lib/umake/gcc-7.5/x86_64/release/libUtilities.a: source/Lib/Utilities/CMakeFiles/Utilities.dir/program_options_lite.cpp.o
/S3/sty/STRANet_folder/VVCSoftware_VTM-VTM-10.2-fast/lib/umake/gcc-7.5/x86_64/release/libUtilities.a: source/Lib/Utilities/CMakeFiles/Utilities.dir/build.make
/S3/sty/STRANet_folder/VVCSoftware_VTM-VTM-10.2-fast/lib/umake/gcc-7.5/x86_64/release/libUtilities.a: source/Lib/Utilities/CMakeFiles/Utilities.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/S3/sty/STRANet_folder/VVCSoftware_VTM-VTM-10.2-fast/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library /S3/sty/STRANet_folder/VVCSoftware_VTM-VTM-10.2-fast/lib/umake/gcc-7.5/x86_64/release/libUtilities.a"
	cd /S3/sty/STRANet_folder/VVCSoftware_VTM-VTM-10.2-fast/build/source/Lib/Utilities && $(CMAKE_COMMAND) -P CMakeFiles/Utilities.dir/cmake_clean_target.cmake
	cd /S3/sty/STRANet_folder/VVCSoftware_VTM-VTM-10.2-fast/build/source/Lib/Utilities && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Utilities.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
source/Lib/Utilities/CMakeFiles/Utilities.dir/build: /S3/sty/STRANet_folder/VVCSoftware_VTM-VTM-10.2-fast/lib/umake/gcc-7.5/x86_64/release/libUtilities.a
.PHONY : source/Lib/Utilities/CMakeFiles/Utilities.dir/build

source/Lib/Utilities/CMakeFiles/Utilities.dir/clean:
	cd /S3/sty/STRANet_folder/VVCSoftware_VTM-VTM-10.2-fast/build/source/Lib/Utilities && $(CMAKE_COMMAND) -P CMakeFiles/Utilities.dir/cmake_clean.cmake
.PHONY : source/Lib/Utilities/CMakeFiles/Utilities.dir/clean

source/Lib/Utilities/CMakeFiles/Utilities.dir/depend:
	cd /S3/sty/STRANet_folder/VVCSoftware_VTM-VTM-10.2-fast/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /S3/sty/STRANet_folder/VVCSoftware_VTM-VTM-10.2-fast /S3/sty/STRANet_folder/VVCSoftware_VTM-VTM-10.2-fast/source/Lib/Utilities /S3/sty/STRANet_folder/VVCSoftware_VTM-VTM-10.2-fast/build /S3/sty/STRANet_folder/VVCSoftware_VTM-VTM-10.2-fast/build/source/Lib/Utilities /S3/sty/STRANet_folder/VVCSoftware_VTM-VTM-10.2-fast/build/source/Lib/Utilities/CMakeFiles/Utilities.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : source/Lib/Utilities/CMakeFiles/Utilities.dir/depend
