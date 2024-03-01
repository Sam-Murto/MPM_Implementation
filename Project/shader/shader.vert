#version 330 core

uniform vec2 resolution;

#define PARTICLE_SIZE 8.0

layout(location = 0) in vec2 position;
layout(location = 1) in vec3 color;

out vec3 particle_color;

vec2 screen_to_device(vec2 pos){
	return (pos - resolution * 0.5) / (resolution * 0.5);
}

void main(){
	vec2 uv = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1));

	gl_Position = vec4(
		screen_to_device((position + uv) * PARTICLE_SIZE),
		0.0f,
		1.0f
	);

	particle_color = color;
}

void P2G(){
}

void G2P(){
}