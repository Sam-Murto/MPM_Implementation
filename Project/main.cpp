#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <math.h>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

#include "Shader.h"
#include <Eigen/Dense>

#include <nanogui/nanogui.h>

static bool keys[1024];

const float PI = 3.14159265359;

const int SCREEN_WIDTH = 1200;
const int SCREEN_HEIGHT = 1000;

//Used for rendering
enum VertexAttributes {
	POSITION_ATTRIBUTE,
	COLOR_ATTRIBUTE
};

struct Vertex {
	glm::vec2 pos;
	glm::vec3 color;
	//glm::vec3 pos;
};

const int MAX_VERTICES = 4096;
int vertexCount = 0;
Vertex vertices[MAX_VERTICES];

//Used for simulation
struct Particle {
	glm::vec2 pos;
	glm::vec2 vel;
	glm::mat2 C; //Affine momentum matrix
	glm::mat2 F; //Deformation gradient
	//glm::mat2 Fe; //Elastic deformation gradient
	//glm::mat2 Fp; //Plastic deformation gradient
	//float Jp; //Determinant of F, proportion of current volume to initial
	float mass;
	float initialVolume;
};

int numParticles = 0;
Particle particles[MAX_VERTICES];

struct Cell {
	glm::vec2 v;
	float mass;
	
};

const int GRID_RESOLUTION = 128;
Cell grid[GRID_RESOLUTION][GRID_RESOLUTION];

glm::vec2 weights[3];
//glm::vec3 weights[3];

//Keep timeStep low, large time steps will result in undefined calculations and crash the program
const GLfloat timeStep = 0.005f;

float hardening = 4.0f;

//MPM Course Pg 19, EQ 47, E is elasticity of material, nu is deformation of material
// Desired E for snow is E = 1.5e5, but that breaks the program without an incredibly low timestep, much lower than other implementations
float E = 1.0e3f;
float nu = 0.3f;


float initialMu = E / (2 * (1 + nu));
float initialLam = (E * nu) / ((1 + nu) * (1 - 2 * nu));

const float gravity = -0.3f;

GLuint vao = 0;
GLuint vbo = 0;

//Nanogui
nanogui::Screen *screen;

float randomFloat() {
	return (float)rand() / (float)RAND_MAX;
}

float randomFloat(float min, float max) {
	//Random 0 to 1 float
	float random = randomFloat();

	//conver 0 to 1 to min to max
	random *= max - min;
	random += min;
	return random;
}

//Planning to switch entirely to Eigen for the SVD functionality without needing to convert
glm::mat2 eigenToGLM(Eigen::Matrix2d eMat) {

	glm::mat2 mat = glm::mat2(1.0f);

	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 2; j++)
			mat[j][i] = eMat(i, j);

	return mat;
}

Eigen::Matrix2d glmToEigen(glm::mat2 gMat) {

	Eigen::Matrix2d mat;

	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 2; j++)
			mat(i,j) = gMat[j][i];

	return mat;
}

void initializeBuffers(void) {
	//Bind vao and vbo
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	glBufferData(GL_ARRAY_BUFFER, MAX_VERTICES * sizeof(vertices[0]), vertices, GL_DYNAMIC_DRAW);

	//Enable all the vertex attributes
	glEnableVertexAttribArray(POSITION_ATTRIBUTE);
	glVertexAttribPointer(POSITION_ATTRIBUTE, 2, GL_FLOAT, GL_FALSE, sizeof(vertices[0]), (GLvoid*) 0);
	glVertexAttribDivisor(POSITION_ATTRIBUTE, 1);

	glEnableVertexAttribArray(COLOR_ATTRIBUTE);
	glVertexAttribPointer(COLOR_ATTRIBUTE, 3, GL_FLOAT, GL_FALSE, sizeof(vertices[0]), (GLvoid*)(sizeof(vertices[0].pos)));
	glVertexAttribDivisor(COLOR_ATTRIBUTE, 1);



}

//info that will be pushed to the shader
void pushVertex(glm::vec2 pos, glm::vec3 color) {
	vertices[vertexCount] = Vertex{ pos, color };
	vertexCount++;
}

void clearBuffers(void) {
	vertexCount = 0;
}

void syncBuffers(void) {
	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferSubData(GL_ARRAY_BUFFER, 0, vertexCount * sizeof(vertices[0]), vertices);
}

void setupUniformValues(Shader& shader) {

	glUniform2f(glGetUniformLocation(shader.program, "resolution"), SCREEN_WIDTH, SCREEN_HEIGHT);
	
}

void resizeWindow(GLFWwindow* window, int width, int height) {
	(void) window;
	glViewport(width / 2 - SCREEN_WIDTH / 2, height / 2 - SCREEN_HEIGHT / 2, SCREEN_WIDTH, SCREEN_HEIGHT);
}

void createSquare(glm::vec2 center, float sideLength, glm::vec2 initialVelocity, int particlesToAdd) {

	float x = 0;
	float y = 0;


	for (int i = 0; i < particlesToAdd; i++) {
		x = randomFloat(-sideLength, sideLength) + center.x;
		y = randomFloat(-sideLength, sideLength) + center.y;
		assert((x > 1) && (x < GRID_RESOLUTION - 1) && ("Circle out of bounds"));
		assert((y > 1) && (y < GRID_RESOLUTION - 1) && ("Circle out of bounds"));
		assert((numParticles < MAX_VERTICES) && "Too many particles");

		particles[numParticles] = Particle{ {x, y}, initialVelocity, glm::mat2(0.0f), glm::mat2(1.0f), 1.0f, 0.0f };
		numParticles++;

	}

}

void createCircle(glm::vec2 center, float radius, glm::vec2 initialVelocity, int particlesToAdd) {

	float theta = 0;
	float r = 0;

	for (int i = 0; i < particlesToAdd; i++) {
		theta = randomFloat(0, 2 * PI);
		r = randomFloat(0, radius);
		float x = center.x + cos(theta) * r;
		float y = center.y + sin(theta) * r;

		particles[numParticles] = Particle{ {x, y}, initialVelocity, glm::mat2(0.0f), glm::mat2(1.0f), 1.0f, 0.0f };
		numParticles++;
	}

}

void initializeParticles() {
	numParticles = 0;

	createCircle({ 16, 32 }, 10.0f, { 12.0f, 3.0f }, 100);
	createCircle({ 100, 32 }, 10.0f, { -8.0f, 5.0f }, 100);
	//createCircle({ 64, 64 }, 20.0f, { 0.0f, 10.0f }, 1000);
}

void initializeGrid() {
	for (int x = 0; x < GRID_RESOLUTION; x++) {
		for (int y = 0; y < GRID_RESOLUTION; y++) {
				grid[x][y] = Cell{ glm::vec2(0), 0 };

		}

	}
}

void initializeVerticies() {
	for (int i = 0; i < numParticles; i++)
	{
		pushVertex(particles[i].pos, glm::vec3(1.0f, 1.0f, 1.0f));
	}
}

void polarDecomposition(glm::mat2 mat, glm::mat2 r, glm::mat2 s) {



}

//Incremental MPM, the 88 line version uses 1.5f - cellDiff for weights[0]. Doing that breaks my simulation, though I'm not sure why.
void calculateWeights(glm::uvec2 cellIndex, glm::vec2 pos) {

	glm::vec2 cellDiff = pos - (glm::vec2)cellIndex - 0.5f;
	weights[0].x = 0.5f * glm::pow(0.5f - cellDiff.x, 2);
	weights[0].y = 0.5f * glm::pow(0.5f - cellDiff.y, 2);
	weights[1].x = 0.75f - glm::pow(cellDiff.x, 2);
	weights[1].y = 0.75f - glm::pow(cellDiff.y, 2);
	weights[2].x = 0.5f * glm::pow(0.5f + cellDiff.x, 2);
	weights[2].y = 0.5f * glm::pow(0.5f + cellDiff.y, 2);

}

//Transfers all particle data to the grid: Incremental MPM
void P2G() {

	for (int i = 0; i < numParticles; i++) {
		Particle p = particles[i];

		glm::mat2 stress = glm::mat2(0.0f);

		glm::mat2 f = p.F;
		
		float j = glm::determinant(f);

		float volume = p.initialVolume * j;


		//Do this for Fe
		//SVD of F
		//Eigen::JacobiSVD<Eigen::Matrix2d, 2> svd(glmToEigen(f));
		//glm::mat2 u = eigenToGLM(svd.matrixU());
		//glm::mat2 v = eigenToGLM(svd.matrixV());
		//Eigen::Vector2d eS = svd.singularValues();
		//glm::mat2 sigma = glm::mat2(1.0f);
		//sigma[0][0] = eS(0);
		//sigma[1][1] = eS(1);

		//Get polar decomp
		//glm::mat2 r = u * glm::transpose(v);
		//glm::mat2 s = v * sigma * glm::transpose(v);
		
		float e = std::exp(hardening * (1.0f - j));
		float eMu = initialMu * e;
		float eLam = initialLam * e;


		glm::mat2 ft = glm::transpose(f);
		glm::mat2 fit = glm::inverse(ft);

		//First Piola-Kirchoff Stress: MPM Course pg 19, 48
		glm::mat2 pkStress = eMu * (f - fit) + eLam * glm::log(j) * fit;
		//glm::mat2 pkStress = 2 * eMu * (f - r) + eLam * (j - 1) * j * fit;

		//Cauchy stress
		stress = (1.0f / j) * ( pkStress * ft );

		glm::mat2 eq0 = -volume * 4.0f * stress * timeStep;

		// Calculate quadratic interpolation weights
		glm::uvec2 cellIndex = (glm::uvec2)p.pos;
		calculateWeights(cellIndex, p.pos);

		//Loop through surrounding cells, Incremental MPM
		for (glm::uint gx = 0; gx < 3; ++gx) {
			for (glm::uint gy = 0; gy < 3; ++gy) {
				float weight = weights[gx].x * weights[gy].y;

				glm::uvec2 neighborCellIndex = glm::uvec2(cellIndex.x + gx - 1, cellIndex.y + gy - 1);
				glm::vec2 cellDist = ((glm::vec2)neighborCellIndex - p.pos) + 0.5f;
				glm::vec2 Q = p.C * cellDist;

				Cell cell = grid[neighborCellIndex.x][neighborCellIndex.y];
				
				// Contribute mass to surrounding cells
				float mass_contrib = weight * p.mass;
				cell.mass += mass_contrib;

				// Adds momentum to the grid to be converted into velocity when the grid is updated
				cell.v += mass_contrib * (p.vel + Q);

				glm::vec2 momentum = (eq0 * weight) * cellDist;
				
				cell.v += momentum;

				grid[neighborCellIndex.x][neighborCellIndex.y] = cell;

			}
		}

	}

}

//Convert grid data to particle data
void G2P() {
	for (int i = 0; i < numParticles; i++) {

		Particle p = particles[i];

		// Reset velocity
		p.C = glm::mat2(0.0f);
		p.vel = glm::vec2(0.0f);

		// Calculate quadratic interpolation weights
		glm::uvec2 cellIndex = (glm::uvec2)p.pos;
		calculateWeights(cellIndex, p.pos);

		glm::mat2 B = glm::mat2(0.0f);

		for (glm::uint gx = 0; gx < 3; ++gx) {
			for (glm::uint gy = 0; gy < 3; ++gy) {
				float weight = weights[gx].x * weights[gy].y;

				glm::uvec2 neighborCellIndex = glm::uvec2(cellIndex.x + gx - 1, cellIndex.y + gy - 1);

				glm::vec2 dist = ((glm::vec2)neighborCellIndex - p.pos) + 0.5f;

				glm::vec2 weighted_velocity = grid[neighborCellIndex.x][neighborCellIndex.y].v * weight;

				glm::mat2 term = glm::mat2(weighted_velocity * dist.x, weighted_velocity * dist.y);

				B += term;

				p.vel += weighted_velocity;
			}
		}

		//Update momentum matrix of particle
		p.C = B * 4.0f;

		//Update position of particle
		p.pos += p.vel * timeStep;

		//Constrain particle positions to stay inside the grid area
		if (p.pos.x < 1)
			p.pos.x = 1;
		if (p.pos.x > GRID_RESOLUTION - 2)
			p.pos.x = GRID_RESOLUTION - 2;
		if (p.pos.y < 1)
			p.pos.y = 1;
		if (p.pos.y > GRID_RESOLUTION - 2)
			p.pos.y = GRID_RESOLUTION - 2;

		glm::mat2 f = glm::mat2(1.0f);
		f += timeStep * p.C;
		p.F = f * p.F;

		//Update particle
		particles[i] = p;
	

	}
}

void simulate() {

	//Reset grid
	initializeGrid();
	// Convert particle data to grid data
	P2G();

	//Update grid
	for (int i = 0; i < GRID_RESOLUTION; i++)
	{
		for (int j = 0; j < GRID_RESOLUTION; j++)
		{
			Cell cell = grid[i][j];

			if (cell.mass > 0)
			{
				cell.v /= cell.mass;
				cell.v += timeStep * glm::vec2(0, gravity);

				if (i < 2 || i > GRID_RESOLUTION - 3) 
					cell.v.x = 0.0f;
				if (j < 2 || j > GRID_RESOLUTION - 3) 
					cell.v.y = 0.0f;
			}

			grid[i][j] = cell;

		}
	}
	// Convert grid data to particle data
	G2P();
}

void calculateVolume() {

	for (int i = 0; i < numParticles; i++) {
		Particle p = particles[i];
		glm::uvec2 cellIndex = (glm::uvec2)p.pos;

		calculateWeights(cellIndex, p.pos);

		float density = 0.0f;

		for (int gx = 0; gx < 3; gx++) {
			for (int gy = 0; gy < 3; gy++)
			{

				float weight = weights[gx].x * weights[gy].y;
				
				glm::uvec2 neighborCellIndex = glm::uvec2(cellIndex.x + gx - 1, cellIndex.y + gy - 1);
				density += grid[neighborCellIndex.x][neighborCellIndex.y].mass * weight;
				

			}
		}

		float volume = p.mass / density;
		
		p.initialVolume = volume;
		particles[i] = p;

	}

}

void initializeNanoGUI(GLFWwindow *window) {

	int width, height;
	glfwGetFramebufferSize(window, &width, &height);
	glViewport(0, 0, width, height);
	glfwSwapInterval(0);
	glfwSwapBuffers(window);

	screen = new nanogui::Screen();
	screen->initialize(window, true);

	nanogui::FormHelper* gui_1 = new nanogui::FormHelper(screen);

	nanogui::ref<nanogui::Window> nanoguiWindow_1 = gui_1->addWindow(Eigen::Vector2i(0, 0), "Nanogui control bar_1");

	screen->setVisible(true);
	screen->performLayout();
	
}

int main()
{
	srand((unsigned)glfwGetTime());
	

	//Initialize glfw
	if (!glfwInit())
	{
		std::cout << "ERROR: GLFW could not initialize" << std::endl;
		return 1;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	//Create window
	GLFWwindow* const window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "MPM", NULL, NULL);


	if (window == NULL) {
		std::cout << "ERROR: Could not create window" << std::endl;
		glfwTerminate();
		return 1;
	}

	glfwMakeContextCurrent(window);

	glewExperimental = GL_TRUE;
	if (GLEW_OK != glewInit()) {
		std::cout << "ERROR: Could not initialize glew" << std::endl;
		return 1;
	}

	glfwSetFramebufferSizeCallback(window, resizeWindow);

	//Get keyboard input
	glfwSetKeyCallback(window,
		[](GLFWwindow* window, int key, int scancode, int action, int mods) {
			//screen->keyCallbackEvent(key, scancode, action, mods);

			if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
				glfwSetWindowShouldClose(window, GL_TRUE);
			if (key >= 0 && key < 1024)
			{
				if (action == GLFW_PRESS)
					keys[key] = true;
				else if (action == GLFW_RELEASE)
					keys[key] = false;
			}
		}
	);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	//NanoGUI
	initializeNanoGUI(window);

	//Get the shader
	Shader shader = Shader("./shader/shader.vert", "./shader/shader.frag");

	initializeBuffers();

	initializeParticles();

	//Initial volumes
	P2G();
	calculateVolume();

	bool begin = false;
	bool keyHeld = false;

	float currentFrame = 0.0f;
	float lastFrame = 0.0f;
	float deltaTime = 0.0f;

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		
		currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;


		if (keys[GLFW_KEY_SPACE]) {
			begin = true;
		}

		if (keys[GLFW_KEY_R]) {
			begin = false;
			initializeParticles();
			initializeGrid();
			P2G();
			calculateVolume();
		}

		if (begin) {
			int numIterations = 0.4f / timeStep;
			if (numIterations < 1)
				numIterations = 1;
			for(int i = 0; i < numIterations; i++)
				simulate();
		}

		// Stuff for verticies & drawing
		clearBuffers();
		initializeVerticies();
		syncBuffers();

		glClear(GL_COLOR_BUFFER_BIT);



		glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, vertexCount);

		shader.use();
		setupUniformValues(shader);

		//screen->drawWidgets();

		glfwSwapBuffers(window);

	}

	return 0;
}