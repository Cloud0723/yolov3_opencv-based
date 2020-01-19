cmake_minimum_required(VERSION 2.8)
project( map_calibrate )

find_package( OpenCV REQUIRED )

# for arm
#include_directories( include ${OpenCV_INCLUDE_DIRS} /opt/MVS/include )
#link_directories(/opt/MVS/lib/aarch64)

# for x86_64
include_directories( include ${OpenCV_INCLUDE_DIRS} /opt/MVS/include)
#link_directories(/opt/MVS/lib/64)
link_directories(/opt/MVS/lib/64)
aux_source_directory(. DIR_SRCS)

#aux_source_directory( tinyxml/ TinyXML_SRCS )

# add_executable( map_calibrate map_calibrate.cc ${TinyXML_SRCS} )
# target_link_libraries( map_calibrate ${OpenCV_LIBS} )

add_executable( object_detection object_detection.cc ${DIR_SRCS} )
target_link_libraries( object_detection MvCameraControl ${OpenCV_LIBS} )