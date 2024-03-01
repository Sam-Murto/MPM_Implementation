#version 330 core

/* Takes in particle color from vertext shader */
in vec3 particle_color;

void main()
{
	gl_FragColor = vec4(particle_color, 1.0);
}