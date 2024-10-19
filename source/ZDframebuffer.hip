	// ZDframebuffer.cpp
#include "../include/ZDframebuffer.h"

template<unsigned int block_size>
__global__ 
void zero_buffs(color_t* c_buff, float* d_buff, int_t width, int_t height) {
	int j = blockDim.y * blockIdx.y + threadIdx.y,
		i = blockDim.x * blockIdx.x + threadIdx.x,
		x = (j * 128 + i) % width,
		y = ((j * 128 + i) - x) / width;
	uint32_t idx = y * width + x;

	c_buff[idx] = color_t{ 0.0f, 0.0f, 0.0f, 1.0f };
	d_buff[idx] = 0.0f;
}

void copy_color_buffer(d_ZDframebuffer* fb, color_t* c_buffer) {

	hip_check(hipMemcpy(c_buffer, fb->color_buffer, sizeof(color_t) * fb->width * fb->height, hipMemcpyDeviceToHost));
	hipDeviceSynchronize();
}

d_ZDframebuffer* create_framebuffer(int_t width, int_t height) {
	d_ZDframebuffer* buff = new d_ZDframebuffer{ nullptr, nullptr, -1, -1 };


	buff->width = width;
	buff->height = height;

	hip_check(hipMalloc((void**)&buff->color_buffer, static_cast<size_t>(buff->width) * buff->height * sizeof(color_t)));
	hip_check(hipMalloc((void**)&buff->depth_buffer, static_cast<size_t>(buff->width) * buff->height * sizeof(float)));

	return buff;
}

void zero_buffers(d_ZDframebuffer* buff) {
	
	color_t* d_color_buff = buff->color_buffer;
	float* d_depth_buff = buff->depth_buffer;

	zero_buffs<block_size>
		<<<buff->height * buff->width / block_size, block_size, 0, hipStreamDefault >>> (d_color_buff, d_depth_buff, buff->width, buff->height);
	hip_check(hipGetLastError());
}

void cleanup_framebuffer(d_ZDframebuffer* buff) {
	hipFree(buff->color_buffer);
	hipFree(buff->depth_buffer);
}