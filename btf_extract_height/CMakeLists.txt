PROJECT(FreeBTF)

IF(MSVC)
    ADD_DEFINITIONS("/DNOMINMAX /D_SCL_SECURE_NO_WARNINGS /D_CRT_SECURE_NO_WARNINGS /arch:AVX")
ELSE()
	ADD_DEFINITIONS("-msse -msse2 -msse3 -mssse3 -msse4 -msse4.1 -msse4.2 -mavx -mf16c")
ENDIF()

ADD_EXECUTABLE(btf-slice btf-slice.cc)
# Build the height-dump tool from the .cpp source
ADD_EXECUTABLE(btf-height btf-height.cpp)
# Also provide the expected CLI name from comments/usage
ADD_EXECUTABLE(btf_dump_height btf-height.cpp)
