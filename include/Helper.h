#ifndef HELPER_H
#define HELPER_H

#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <queue>
#include <stb_image.h>

#include <hip/hip_runtime.h>

#include <GLFW/glfw3.h>

const int block_size = 16;

void hip_check(hipError_t e) {
	if (e != HIP_SUCCESS) {
		std::cerr << "HIP ERROR: " << hipGetErrorString(e) << std::endl;
	}
}

#endif