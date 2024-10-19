#ifndef ZDRENDER_H
#define ZDRENDER_H

#include "ZDlevel.h"

__device__ float line_equation(vec3_t, vec2_t);
__device__ uv_t compute_barycentric(vec2_t, uv_t, uv_t, uv_t);

namespace ZDrender {
	void calculate_instance_visibility(d_ZDmodel*, d_ZDinstance*, int_t, d_ZDcamera*);

	void draw(d_ZDframebuffer*, d_ZDmodel*, d_ZDinstance*, d_ZDcamera*, int_t);
}

#endif
