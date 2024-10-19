	// ZDrender.cpp
#include "../include/ZDrender.h"

class cull_models_from_camera;
class draw_instances;
class flat_shade;

uv_t compute_barycentric(vec2_t p, uv_t a, uv_t b, uv_t c) {
	float denom = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);

	// Calculate barycentric coordinates lambdaA, lambdaB, lambdaC
	float lambdaA = ((b.y - c.y) * (p.x - c.x) + (c.x - b.x) * (p.y - c.y)) / denom;
	float lambdaB = ((c.y - a.y) * (p.x - c.x) + (a.x - c.x) * (p.y - c.y)) / denom;
	float lambdaC = 1.0f - lambdaA - lambdaB;

	// The point's barycentric coordinates (lambdaA, lambdaB, lambdaC) relative to the triangle
	uv_t barycentric;
	barycentric.x = lambdaA;  // Corresponds to the weight for vertex A
	barycentric.y = lambdaB;  // Corresponds to the weight for vertex B
	// You could return lambdaC as well, but since the sum is 1, we skip it here.

	return barycentric;
}

__global__
void cull_models(d_ZDmodel* d_models, d_ZDinstance* d_instances, int_t instance_count, d_ZDcamera* d_camera) {
	int j = blockDim.y * blockIdx.y + threadIdx.y,
		i = blockDim.x * blockIdx.x + threadIdx.x;
	int index = j * 16 + i;

	if (index < instance_count) {
		d_ZDinstance* inst = &d_instances[index];
		d_ZDmodel* model = &d_models[inst->model_index];

		vec3_t cam_to_inst = ZD::subtract_v3(d_camera->position, inst->position);
		float dist = sqrtf(cam_to_inst.x * cam_to_inst.x + cam_to_inst.y * cam_to_inst.y + cam_to_inst.z * cam_to_inst.z);
		float d_prod = ZD::dot(cam_to_inst, d_camera->direction);
		if (d_prod <= 0.0f || dist < 0.1f || dist > 1000.0f) {
			inst->show = false;
		}
		else {
			cam_to_inst = ZD::normalize(cam_to_inst);
			float angle = ZD::dot(d_camera->direction, cam_to_inst);
			if (angle >= cosf(d_camera->hori_fov * 0.5f)) {
				inst->show = true;
			}
			else {
				inst->show = false;
			}
		}
	}
}

void ZDrender::calculate_instance_visibility(d_ZDmodel* models, d_ZDinstance* instances, int_t instance_count, d_ZDcamera* camera) {
	cull_models<<<(instance_count / block_size) + 1, block_size, 0, hipStreamDefault >>>(models, instances, instance_count, camera);
}

__global__
void vertex_shader(d_ZDmodel* d_models, d_ZDinstance* d_instances, int_t instance_count, d_ZDcamera* d_camera, int_t width, int_t height) {
	int j = blockDim.y * blockIdx.y + threadIdx.y,
		i = blockDim.x * blockIdx.x + threadIdx.x,
		x = (j * 16 + i) % width,
		y = ((j * 16 + i) - x) / width;
	int index = y * width + x;

	if (index < instance_count) {
		d_ZDinstance* inst = &d_instances[index];
		d_ZDmodel* model = &d_models[inst->model_index];

		vec3_t* positions = model->vertex_positions;
		vec3_t* normals = model->triangle_normals;
		tri_t* tri_idxs = model->triangle_indices;
		// float fov, float aspectRatio, float nearPlane, float farPlane
		mat4_t perspect = ZD::perspective(d_camera->hori_fov, static_cast<float>(width) / height, 0.01f, 100.0f);
		bool fail_invert = false;
		mat4_t cam_mtx = ZD::invert(ZD::rotate(d_camera->rotation), fail_invert);
		mat4_t rotation = ZD::rotate(inst->rotation);

		for (uint_t i = 0; i < inst->vertex_count; i++) {
			//if (inst->visible_triangles[i]) {
			vec3_t v0 = positions[i];
			vec3_t n0 =
				v0 = ZD::add_v3(v0, ZD::subtract_v3(d_camera->position, inst->position));

			v0 = ZD::to_vec3(ZD::product_m4(rotation, ZD::to_vec4(v0, 0.0f)));
			v0 = ZD::to_vec3(ZD::product_m4(cam_mtx, ZD::to_vec4(v0, 0.0f)));
			v0 = ZD::to_vec3(ZD::product_m4(perspect, ZD::to_vec4(v0, 0.0f)));


			inst->transformed_vertices[i] = vec3_t{ inst->scale * v0.x, inst->scale * v0.y, inst->scale * v0.z };
		}
		for (uint_t i = 0; i < inst->triangle_count; i++) {
			vec3_t v0 = normals[i];
			//v0 = ZD::add_v3(v0, ZD::subtract_v3(d_camera->position, inst->position));

			v0 = ZD::to_vec3(ZD::product_m4(rotation, ZD::to_vec4(v0, 0.0f)));
			//v0 = ZD::to_vec3(ZD::product_m4(cam_mtx, ZD::to_vec4(v0, 0.0f)));


			inst->transformed_normals[i] = v0;
		}
	}
}

__global__
void fragment_shader(d_ZDmodel* d_models, d_ZDinstance* d_instances, int_t instance_count, d_ZDcamera* d_camera, int_t width, int_t height, color_t* d_color_buff, float* d_depth_buff) {
	int j = blockDim.y * blockIdx.y + threadIdx.y,
		i = blockDim.x * blockIdx.x + threadIdx.x,
		x = (j * 16 + i) % width,
		y = ((j * 16 + i) - x) / width;
	int index = y * width + x;

	vec2_t normalized_coord{};

	float ratio = static_cast<float>(width) / static_cast<float>(height);
	float norm_x = (static_cast<float>(x) - (static_cast<float>(width) * 0.5f)) / (static_cast<float>(width));
	float norm_y = (static_cast<float>(y) - (static_cast<float>(height) * 0.5f)) / (static_cast<float>(height));
	float fov_rad = d_camera->hori_fov * (PI / 180.0f);
	float half_fov = fov_rad * 0.5f;

	normalized_coord.x = norm_x;
	normalized_coord.y = norm_y;

	d_color_buff[y * width + x] = color_t{ 0.17f, 0.15f, 0.17f, 1.0f };
	d_ZDvertex_sample* samp = &d_camera->vertex_samples[index];
	samp->depth = 1000.0f;
	samp->hit = false;

	for (uint_t i = 0; i < instance_count; i++) {
		d_ZDinstance* instance = &d_instances[i];
		if (instance->show) {
			d_ZDmodel* model = &d_models[instance->model_index];
			for (uint_t j = 0; j < instance->triangle_count; j++) {
				//if (instance->visible_triangles[j]) {
				tri_t t = model->triangle_indices[j];
				vec3_t normal = instance->transformed_normals[j];

				//float n_dot = ZD::dot(normal, d_camera->direction);
				//if (n_dot <= 0.0f) {

				vec3_t v0 = instance->transformed_vertices[t.a],
					v1 = instance->transformed_vertices[t.b],
					v2 = instance->transformed_vertices[t.c];

				vec3_t min = ZD::max_v3(v0, ZD::max_v3(v1, v2));

				vec3_t depth = ZD::subtract_v3(d_camera->position, min);
				float depth_test = sqrtf(depth.x * depth.x + depth.y * depth.y + depth.z * depth.z);

				if (depth_test < samp->depth) {
					vec2_t v0a = { -v0.x / v0.z, -v0.y / v0.z },
						v1a = { -v1.x / v1.z, -v1.y / v1.z },
						v2a = { -v2.x / v2.z, -v2.y / v2.z };

					float sign1 = ZD::line_equation(normalized_coord, v0a, v1a),
						sign2 = ZD::line_equation(normalized_coord, v1a, v2a),
						sign3 = ZD::line_equation(normalized_coord, v2a, v0a);


					if ((sign1 >= 0.0f && sign2 >= 0.0f && sign3 >= 0.0f) ||
						(sign1 <= 0.0f && sign2 <= 0.0f && sign3 <= 0.0f)) {
						uv_t min = { ZD::minf(v0a.x, ZD::minf(v1a.x, v2a.x)), ZD::minf(v0a.y, ZD::minf(v1a.y, v2a.y)) };
						uv_t max = { ZD::maxf(v0a.x, ZD::minf(v1a.x, v2a.x)), ZD::maxf(v0a.y, ZD::minf(v1a.y, v2a.y)) };
						uv_t comp_coord = uv_t{ (static_cast<float>(x) - min.x) / (max.x - min.x), (static_cast<float>(y) - min.y) / (max.y - min.y) };

						uv_t uv = compute_barycentric(comp_coord, d_models[instance->model_index].vertex_uvs[t.a], d_models[instance->model_index].vertex_uvs[t.b], d_models[instance->model_index].vertex_uvs[t.c]);

						samp->uv_coord = uv;
						samp->hit = true;

						//d_color_buff[y * *d_width + x] = color_t{ depth_test / 100.0f, depth_test / 100.0f, depth_test / 100.0f, 1.0f };
						samp->depth = depth_test;
						samp->model_index = instance->model_index;
						samp->triangle_index = j;
						samp->triangle_normal = normal;
						d_depth_buff[index] = depth_test;
					}
					else {
					}
				}
				//}
			//}
			}
		}
	}
}

__global__
void shade(d_ZDmodel* d_models, d_ZDinstance* d_instances, int_t instance_count, d_ZDcamera* d_camera, int_t width, int_t height, color_t* d_color_buff, float* d_depth_buff) {
	int j = blockDim.y * blockIdx.y + threadIdx.y,
		i = blockDim.x * blockIdx.x + threadIdx.x,
		x = (j * 16 + i) % width,
		y = ((j * 16 + i) - x) / width;
	int index = y * width + x;

	d_ZDvertex_sample* smp = &d_camera->vertex_samples[y * width + x];
	if (smp->hit) {
		vec3_t sun_direction = vec3_t{ 0.10f, 0.33f, 0.2f };
		color_t sun_color = color_t{ 0.72f, 0.45f, 0.93f, 1.0f };
		float sun_intensity = 747.0f;

		float diffuse = ZD::dot(smp->triangle_normal, sun_direction);
		if (diffuse >= 0.0f) {
			d_color_buff[index] = color_t{ sun_intensity * sun_color.x * diffuse, sun_intensity * sun_color.y * diffuse, sun_intensity * sun_color.z * diffuse , sun_intensity * sun_color.w * diffuse };
			//d_color_buff[y * *d_width + x] = color_t{ 1.0f, 1.0f, 1.0f, 1.0f };
		}
		else {
			d_color_buff[y * width + x] = color_t{ 0.0f, 0.0f, 0.0f, 1.0f };
		}
	}
}

void ZDrender::draw(d_ZDframebuffer* buff, d_ZDmodel* models, d_ZDinstance* instances, d_ZDcamera* camera, int_t instance_count) {
	color_t* d_color_buff = buff->color_buffer;
	float* d_depth_buff = buff->depth_buffer;
	d_ZDmodel* d_models = models;
	d_ZDinstance* d_instances = instances;
	d_ZDcamera* d_camera = camera;
	int_t* d_width, * d_height, * d_instance_count;

	hip_check(hipMalloc((void**)&d_width, sizeof(int_t)));
	hip_check(hipMalloc((void**)&d_height, sizeof(int_t)));
	hip_check(hipMalloc((void**) & d_instance_count, sizeof(int_t)));

	hip_check(hipMemcpy(d_width, &buff->width, sizeof(int_t), hipMemcpyHostToDevice));
	hip_check(hipMemcpy(d_height, &buff->height, sizeof(int_t), hipMemcpyHostToDevice));
	hip_check(hipMemcpy(d_instance_count, &d_instance_count, sizeof(int_t), hipMemcpyHostToDevice));

	vertex_shader<<<(instance_count / block_size) + 1, block_size, 0, hipStreamDefault >>>(d_models, d_instances, instance_count, d_camera, buff->width, buff->height);
	hipDeviceSynchronize();

	fragment_shader<<< (buff->height * buff->width / block_size), block_size, 0, hipStreamDefault >>>(d_models, d_instances, instance_count, d_camera, buff->width, buff->height, d_color_buff, d_depth_buff);
	hipDeviceSynchronize();

	shade<<< (buff->height * buff->width / block_size), block_size, 0, hipStreamDefault >>>(d_models, d_instances, instance_count, d_camera, buff->width, buff->height, d_color_buff, d_depth_buff);
	hipDeviceSynchronize();

	hip_check(hipFree(d_width));
	hip_check(hipFree(d_height));
	hip_check(hipFree(d_instance_count));

}