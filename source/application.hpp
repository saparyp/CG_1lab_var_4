#pragma once

#include "graphics_internal.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <string>

namespace application {

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    
    static VkVertexInputBindingDescription getBindingDescription();
    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions();
};

struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

// Вершины усеченного тетраэдра
extern std::vector<Vertex> vertices;
// Индексы для усеченного тетраэдра
extern std::vector<uint32_t> indices;

// Vulkan объекты
extern VkPipeline pipeline;
extern VkPipelineLayout pipelineLayout;
extern VkDescriptorSetLayout descriptorSetLayout;
extern VkDescriptorSet descriptorSet;
extern VkBuffer vertexBuffer;
extern VkBuffer indexBuffer;
extern VmaAllocation vertexBufferAllocation;
extern VmaAllocation indexBufferAllocation;
extern VkBuffer uniformBuffer;
extern VmaAllocation uniformBufferAllocation;
extern void* uniformBufferMapped;

// Параметры для UI
extern float rotationSpeed;
extern float scale;
extern bool usePerspective;
extern float fov;
extern glm::vec3 rotation;

bool initialize();
void shutdown();
void update(double time);
void render(const graphics::internal::FrameData& fd);

// Вспомогательные функции
void createPipeline();
void createDescriptorSetLayout();
void createDescriptorSet();
void createBuffers();
void createUniformBuffer();
void updateUniformBuffer();
void createTruncatedTetrahedron();

} // namespace application