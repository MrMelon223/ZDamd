	// ZDmodel.cpp
#include "../include/ZDmodel.h"

void ZDmodel::load_from(std::string file_path) {
	this->file_path = file_path;
	//this->name = extract_name(this->filepath);

	std::ifstream in;
	in.open(this->file_path, std::ios_base::in);

	if (!in) {
		std::cout << "Cannot find model " << file_path << std::endl;
		return;
	}

	size_t len;
	std::string line2;
	std::getline(in, line2);
	std::istringstream in_s(line2);
	in_s >> len;
	for (int i = 0; i < len; i++) {

		std::string type;
		std::getline(in, type);
		std::istringstream in2(type);

		float x, y, z;
		float r, g, b, a;
		float u, v;
		in2 >> x >> y >> z;
		in2 >> r >> g >> b >> a;
		in2 >> u >> v;

		this->vertex_positions.push_back(vec3_t{ x, y, z });
		vec3_t position{ x,y,z };
		position = ZD::normalize(position);
		this->vertex_normals.push_back(position);
		this->vertex_colors.push_back(color_t{ r, g, b, a });
		this->vertex_uvs.push_back(uv_t{ u, v });
	}
	std::string line;
	std::getline(in, line);
	std::istringstream in_s2(line);
	in_s2 >> len;

	for (int i = 0; i < len; i++) {
		std::string type;
		std::getline(in, type);
		std::istringstream in3(type);

		uint32_t x, y, z;
		in3 >> x >> y >> z;

		//std::cout << "Triangle idxs = { " << x << ", " << y << ", " << z << " }" << std::endl;

		vec3_t va = this->vertex_positions[x],
			vb = this->vertex_positions[y],
			vc = this->vertex_positions[z];

		vec3_t op_a = ZD::subtract_v3(vb, va),
			op_b = ZD::subtract_v3(vb, vc);

		vec3_t t_norm = ZD::cross(op_a, op_b),
			pos_a = this->vertex_normals[x],
			pos_b = this->vertex_normals[y],
			pos_c = this->vertex_normals[z],
			op_c = ZD::add_v3(pos_a, pos_b),
			norm_w = ZD::add_v3(op_c, pos_c);

		norm_w.x = norm_w.x / 3.0f;
		norm_w.y = norm_w.y / 3.0f;
		norm_w.z = norm_w.z / 3.0f;

		if (ZD::dot(t_norm, norm_w) < 0.0f) {
			t_norm.x = -t_norm.x;
			t_norm.y = -t_norm.y;	// This is gonna be a major floating point rounding headache in the future, goodluck homie
			t_norm.z = -t_norm.z;
		}

		this->triangle_indices.push_back(tri_t{ x, y, z });
		this->triangle_normals.push_back(t_norm);
	}

	std::string shader_name;
	std::getline(in, line);
	std::istringstream in_s3(line);
	in_s3 >> shader_name;

	//this->shader_program = load_shader_program(shader_name);


	std::cout << std::setw(10) << this->vertex_positions.size() << " Vertices loaded" << std::endl;
	std::cout << std::setw(10) << this->triangle_indices.size() << " Triangles loaded" << std::endl;

	in.close();
}

ZDmodel::ZDmodel(std::string file_path, std::string name) {
	this->name = name;
	this->load_from(file_path);
}

d_ZDmodel ZDmodel::to_gpu() {

	d_ZDmodel ret{};

	hip_check(hipMalloc((void**)&ret.vertex_positions, sizeof(vec3_t) * this->vertex_positions.size()));
	hip_check(hipMalloc((void**)&ret.vertex_normals, sizeof(vec3_t) * this->vertex_normals.size()));
	hip_check(hipMalloc((void**)&ret.vertex_colors, sizeof(color_t) * this->vertex_colors.size()));
	hip_check(hipMalloc((void**)&ret.vertex_uvs, sizeof(uv_t) * this->vertex_uvs.size()));
	hip_check(hipMalloc((void**)&ret.vertex_count, sizeof(int)));

	unsigned int v_count = static_cast<unsigned int>(this->vertex_positions.size()),
		t_count = static_cast<unsigned int>(this->triangle_indices.size());

	hip_check(hipMemcpy(ret.vertex_positions, this->vertex_positions.data(), sizeof(vec3_t) * this->vertex_positions.size(), hipMemcpyHostToDevice));
	hip_check(hipMemcpy(ret.vertex_normals, this->vertex_normals.data(), sizeof(vec3_t) * this->vertex_normals.size(), hipMemcpyHostToDevice));
	hip_check(hipMemcpy(ret.vertex_colors, this->vertex_colors.data(), sizeof(color_t) * this->vertex_colors.size(), hipMemcpyHostToDevice));
	hip_check(hipMemcpy(ret.vertex_uvs, this->vertex_uvs.data(), sizeof(uv_t) * this->vertex_uvs.size(), hipMemcpyHostToDevice));
	hip_check(hipMemcpy(ret.vertex_count, &v_count, sizeof(unsigned int), hipMemcpyHostToDevice));

	hip_check(hipMalloc((void**)&ret.triangle_indices, sizeof(tri_t) * this->triangle_indices.size()));
	hip_check(hipMalloc((void**)&ret.triangle_normals, sizeof(vec3_t) * this->triangle_normals.size()));
	hip_check(hipMalloc((void**)&ret.triangle_count, sizeof(unsigned int)));

	hip_check(hipMemcpy(ret.triangle_indices, this->triangle_indices.data(), sizeof(tri_t) * this->triangle_indices.size(), hipMemcpyHostToDevice));
	hip_check(hipMemcpy(ret.triangle_normals, this->triangle_normals.data(), sizeof(vec3_t) * this->triangle_normals.size(), hipMemcpyHostToDevice));
	hip_check(hipMemcpy(ret.triangle_count, &t_count, sizeof(unsigned int), hipMemcpyHostToDevice));

	hipDeviceSynchronize();

	return ret;
}

d_ZDinstance create_instance(int_t mod_idx, vec3_t position, vec3_t rotation, uint_t v_count, uint_t t_count, bool show, float scale) {
	d_ZDinstance inst{ mod_idx, position, rotation, v_count, t_count, show, scale, nullptr, nullptr, nullptr };

	hip_check(hipMalloc((void**)&inst.visible_triangles, sizeof(bool) * t_count));
	hip_check(hipMalloc((void**)&inst.transformed_vertices, sizeof(vec3_t) * v_count));
	hip_check(hipMalloc((void**)&inst.transformed_normals, sizeof(vec3_t) * t_count));

	hipDeviceSynchronize();

	return inst;
}

void free_instance(d_ZDinstance* inst) {
	hip_check(hipFree(inst->visible_triangles));
	hip_check(hipFree(inst->transformed_vertices));
	hip_check(hipFree(inst->transformed_normals));
	hipDeviceSynchronize();
}