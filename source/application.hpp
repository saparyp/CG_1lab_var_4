#pragma once

#include "graphics_internal.hpp"

#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <vector>

namespace application {

struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    
    static VkVertexInputBindingDescription getBindingDescription();
    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions();
};

extern std::vector<Vertex> vertices;
extern std::vector<uint32_t> indices;

extern VkPipeline pipeline;
extern VkPipelineLayout pipelineLayout;
extern VkBuffer vertexBuffer;
extern VkBuffer indexBuffer;
extern VmaAllocation vertexBufferAllocation;
extern VmaAllocation indexBufferAllocation;

bool initialize();
void shutdown();
void update(double time);
void render(const graphics::internal::FrameData& fd);
void buildImGui();

} // namespace application