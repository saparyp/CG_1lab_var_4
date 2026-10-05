#pragma once

#include "graphics_internal.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <string>
#include <array>

namespace application {

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    
    static VkVertexInputBindingDescription getBindingDescription();
    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions();
};

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::vec4 baseColor;
};

extern std::vector<Vertex> vertices;
extern std::vector<uint32_t> indices;

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

extern float rotationSpeed;
extern float scale;
extern bool usePerspective;
extern float fov;
extern glm::vec3 rotation;

bool initialize();
void shutdown();
void update(double time);
void render(const graphics::internal::FrameData& fd);
void buildImGui();

} // namespace application