#include "application.hpp"

#include <imgui.h>
#include <vk_mem_alloc.h>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace application {
namespace {

struct alignas(16) SceneUniformBuffer {
    glm::mat4 viewProjection;
    glm::vec4 cameraPosition;
    glm::vec4 ambientColorIntensity;
    glm::vec4 directionalDirection;
    glm::vec4 directionalColorIntensity;
    glm::uvec4 lightCounts;
};

struct alignas(16) ModelUniformBuffer {
    glm::mat4 model;
    glm::vec4 albedo;
    glm::vec4 specularColor;
    glm::vec4 materialParameters;
};

struct alignas(16) PointLightGpu {
    glm::vec4 position;
    glm::vec4 colorIntensity;
};

struct alignas(16) SpotLightGpu {
    glm::vec4 position;
    glm::vec4 direction;
    glm::vec4 colorIntensity;
    glm::vec4 coneCosines;
};

static_assert(sizeof(SceneUniformBuffer) == 144);
static_assert(sizeof(ModelUniformBuffer) == 112);
static_assert(sizeof(PointLightGpu) == 32);
static_assert(sizeof(SpotLightGpu) == 64);

struct PointLight {
    glm::vec3 position{2.0f, 2.0f, 2.0f};
    glm::vec3 color{1.0f, 0.75f, 0.45f};
    float intensity{18.0f};
};

struct SpotLight {
    glm::vec3 position{-2.0f, 2.5f, 2.0f};
    glm::vec3 direction{0.4f, -0.5f, -1.0f};
    glm::vec3 color{0.55f, 0.7f, 1.0f};
    float intensity{30.0f};
    float innerAngle{12.0f};
    float outerAngle{22.0f};
};

struct MappedStorageBuffer {
    VkBuffer buffer{VK_NULL_HANDLE};
    VmaAllocation allocation{VK_NULL_HANDLE};
    void* mapped{nullptr};
    VkDeviceSize capacity{0};
};

VkDescriptorSetLayout sceneDescriptorSetLayout{VK_NULL_HANDLE};
VkDescriptorSetLayout modelDescriptorSetLayout{VK_NULL_HANDLE};
VkDescriptorPool descriptorPool{VK_NULL_HANDLE};
VkDescriptorSet sceneDescriptorSet{VK_NULL_HANDLE};
VkDescriptorSet modelDescriptorSet{VK_NULL_HANDLE};

VkBuffer sceneUniformBuffer{VK_NULL_HANDLE};
VmaAllocation sceneUniformBufferAllocation{VK_NULL_HANDLE};
void* sceneUniformBufferMapped{nullptr};
VkBuffer modelUniformBuffer{VK_NULL_HANDLE};
VmaAllocation modelUniformBufferAllocation{VK_NULL_HANDLE};
void* modelUniformBufferMapped{nullptr};
MappedStorageBuffer pointLightBuffer;
MappedStorageBuffer spotLightBuffer;

glm::vec3 modelPosition{0.0f};
glm::vec3 modelRotation{0.0f};
float modelScale{1.0f};
glm::vec3 cameraPosition{0.0f, 0.0f, 5.0f};
float cameraYaw{-90.0f};
float cameraPitch{0.0f};
float fov{45.0f};
bool usePerspective{true};
float orthographicSize{4.0f};
glm::vec4 albedo{0.7f, 0.28f, 0.12f, 1.0f};
glm::vec3 specularColor{1.0f};
float shininess{48.0f};
glm::vec3 ambientColor{0.16f};
float ambientIntensity{0.35f};
glm::vec3 directionalDirection{-0.4f, -1.0f, -0.5f};
glm::vec3 directionalColor{1.0f, 0.95f, 0.82f};
float directionalIntensity{1.0f};
std::vector<PointLight> pointLights{{}};
std::vector<SpotLight> spotLights{{}};
bool isAnimating{false};
float animationSpeed{1.0f};
float trajectoryRadius{2.0f};
float animationTime{0.0f};

VkDevice device() {
    return graphics::internal::context.device;
}

void checkVk(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with VkResult " +
                                 std::to_string(result));
    }
}

std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + filename);
    }
    const auto fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(fileSize));
    if (!file) {
        throw std::runtime_error("Failed to read shader file: " + filename);
    }
    return buffer;
}

VkShaderModule createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule shaderModule{VK_NULL_HANDLE};
    checkVk(vkCreateShaderModule(device(), &createInfo, nullptr, &shaderModule),
            "vkCreateShaderModule");
    return shaderModule;
}

void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage,
                  VkBuffer& buffer, VmaAllocation& allocation) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = memoryUsage;
    checkVk(vmaCreateBuffer(graphics::internal::context.allocator, &bufferInfo, &allocationInfo,
                            &buffer, &allocation, nullptr),
            "vmaCreateBuffer");
}

void createMappedBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                        VkBuffer& buffer, VmaAllocation& allocation, void*& mapped) {
    createBuffer(size, usage, VMA_MEMORY_USAGE_CPU_TO_GPU, buffer, allocation);
    checkVk(vmaMapMemory(graphics::internal::context.allocator, allocation, &mapped),
            "vmaMapMemory");
}

glm::vec3 faceNormal(const std::vector<glm::vec3>& face) {
    glm::vec3 normal = glm::normalize(glm::cross(face[1] - face[0], face[2] - face[0]));
    glm::vec3 center{0.0f};
    for (const auto& vertex : face) {
        center += vertex;
    }
    center /= static_cast<float>(face.size());
    if (glm::dot(normal, center) < 0.0f) {
        normal = -normal;
    }
    return normal;
}

void createTruncatedTetrahedron() {
    const glm::vec3 v0{1.0f, 1.0f, 1.0f};
    const glm::vec3 v1{1.0f, -1.0f, -1.0f};
    const glm::vec3 v2{-1.0f, 1.0f, -1.0f};
    const glm::vec3 v3{-1.0f, -1.0f, 1.0f};
    const auto lerp = [](glm::vec3 a, glm::vec3 b, float t) { return a + (b - a) * t; };

    const glm::vec3 e01a = lerp(v0, v1, 1.0f / 3.0f), e01b = lerp(v0, v1, 2.0f / 3.0f);
    const glm::vec3 e02a = lerp(v0, v2, 1.0f / 3.0f), e02b = lerp(v0, v2, 2.0f / 3.0f);
    const glm::vec3 e03a = lerp(v0, v3, 1.0f / 3.0f), e03b = lerp(v0, v3, 2.0f / 3.0f);
    const glm::vec3 e12a = lerp(v1, v2, 1.0f / 3.0f), e12b = lerp(v1, v2, 2.0f / 3.0f);
    const glm::vec3 e13a = lerp(v1, v3, 1.0f / 3.0f), e13b = lerp(v1, v3, 2.0f / 3.0f);
    const glm::vec3 e23a = lerp(v2, v3, 1.0f / 3.0f), e23b = lerp(v2, v3, 2.0f / 3.0f);

    const auto addFace = [](const std::vector<glm::vec3>& face) {
        const uint32_t first = static_cast<uint32_t>(vertices.size());
        const glm::vec3 normal = faceNormal(face);
        for (const auto& position : face) {
            vertices.push_back({position, normal});
        }
        for (size_t i = 1; i + 1 < face.size(); ++i) {
            indices.push_back(first);
            indices.push_back(first + static_cast<uint32_t>(i));
            indices.push_back(first + static_cast<uint32_t>(i + 1));
        }
    };

    addFace({e01a, e02a, e03a});
    addFace({e01b, e13a, e12a});
    addFace({e02b, e12b, e23a});
    addFace({e03b, e13b, e23b});
    addFace({e01a, e01b, e12a, e12b, e02b, e02a});
    addFace({e01a, e03a, e03b, e13b, e13a, e01b});
    addFace({e02a, e02b, e23a, e23b, e03b, e03a});
    addFace({e12a, e12b, e23a, e23b, e13b, e13a});
}

void createGeometryBuffers() {
    auto& ctx = graphics::internal::context;
    const VkDeviceSize vertexSize = sizeof(vertices[0]) * vertices.size();
    const VkDeviceSize indexSize = sizeof(indices[0]) * indices.size();

    VkBuffer stagingVertexBuffer{VK_NULL_HANDLE}, stagingIndexBuffer{VK_NULL_HANDLE};
    VmaAllocation stagingVertexAllocation{VK_NULL_HANDLE}, stagingIndexAllocation{VK_NULL_HANDLE};
    createBuffer(vertexSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY,
                 stagingVertexBuffer, stagingVertexAllocation);
    createBuffer(indexSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY,
                 stagingIndexBuffer, stagingIndexAllocation);

    void* data{nullptr};
    checkVk(vmaMapMemory(ctx.allocator, stagingVertexAllocation, &data), "vmaMapMemory");
    std::memcpy(data, vertices.data(), static_cast<size_t>(vertexSize));
    checkVk(vmaFlushAllocation(ctx.allocator, stagingVertexAllocation, 0, vertexSize),
            "vmaFlushAllocation");
    vmaUnmapMemory(ctx.allocator, stagingVertexAllocation);
    checkVk(vmaMapMemory(ctx.allocator, stagingIndexAllocation, &data), "vmaMapMemory");
    std::memcpy(data, indices.data(), static_cast<size_t>(indexSize));
    checkVk(vmaFlushAllocation(ctx.allocator, stagingIndexAllocation, 0, indexSize),
            "vmaFlushAllocation");
    vmaUnmapMemory(ctx.allocator, stagingIndexAllocation);

    createBuffer(vertexSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VMA_MEMORY_USAGE_GPU_ONLY, vertexBuffer, vertexBufferAllocation);
    createBuffer(indexSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                 VMA_MEMORY_USAGE_GPU_ONLY, indexBuffer, indexBufferAllocation);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolInfo.queueFamilyIndex = ctx.graphics_queue_index;
    VkCommandPool commandPool{VK_NULL_HANDLE};
    checkVk(vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &commandPool), "vkCreateCommandPool");

    VkCommandBufferAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandPool = commandPool;
    allocateInfo.commandBufferCount = 1;
    VkCommandBuffer commandBuffer{VK_NULL_HANDLE};
    checkVk(vkAllocateCommandBuffers(ctx.device, &allocateInfo, &commandBuffer),
            "vkAllocateCommandBuffers");

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    checkVk(vkBeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer");
    VkBufferCopy copy{};
    copy.size = vertexSize;
    vkCmdCopyBuffer(commandBuffer, stagingVertexBuffer, vertexBuffer, 1, &copy);
    copy.size = indexSize;
    vkCmdCopyBuffer(commandBuffer, stagingIndexBuffer, indexBuffer, 1, &copy);
    checkVk(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    checkVk(vkQueueSubmit(ctx.graphics_queue, 1, &submitInfo, VK_NULL_HANDLE), "vkQueueSubmit");
    checkVk(vkQueueWaitIdle(ctx.graphics_queue), "vkQueueWaitIdle");

    vmaDestroyBuffer(ctx.allocator, stagingVertexBuffer, stagingVertexAllocation);
    vmaDestroyBuffer(ctx.allocator, stagingIndexBuffer, stagingIndexAllocation);
    vkFreeCommandBuffers(ctx.device, commandPool, 1, &commandBuffer);
    vkDestroyCommandPool(ctx.device, commandPool, nullptr);
}

void createDescriptorSetLayouts() {
    const std::array<VkDescriptorSetLayoutBinding, 3> sceneBindings{{
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
    }};
    VkDescriptorSetLayoutCreateInfo sceneInfo{};
    sceneInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    sceneInfo.bindingCount = static_cast<uint32_t>(sceneBindings.size());
    sceneInfo.pBindings = sceneBindings.data();
    checkVk(vkCreateDescriptorSetLayout(device(), &sceneInfo, nullptr, &sceneDescriptorSetLayout),
            "vkCreateDescriptorSetLayout(scene)");

    const VkDescriptorSetLayoutBinding modelBinding{
        0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo modelInfo{};
    modelInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    modelInfo.bindingCount = 1;
    modelInfo.pBindings = &modelBinding;
    checkVk(vkCreateDescriptorSetLayout(device(), &modelInfo, nullptr, &modelDescriptorSetLayout),
            "vkCreateDescriptorSetLayout(model)");
}

void updateStorageDescriptor(uint32_t binding, const MappedStorageBuffer& storage,
                            VkDeviceSize range) {
    VkDescriptorBufferInfo bufferInfo{storage.buffer, 0, range};
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = sceneDescriptorSet;
    write.dstBinding = binding;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &bufferInfo;
    vkUpdateDescriptorSets(device(), 1, &write, 0, nullptr);
}

void createStorageBuffer(MappedStorageBuffer& storage, VkDeviceSize capacity,
                         VkDeviceSize elementSize, uint32_t binding) {
    const VkDeviceSize bufferSize = capacity * elementSize;
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(graphics::internal::context.physical_device, &properties);
    if (bufferSize > properties.limits.maxStorageBufferRange) {
        throw std::runtime_error("Light array exceeds maxStorageBufferRange");
    }

    VkBuffer buffer{VK_NULL_HANDLE};
    VmaAllocation allocation{VK_NULL_HANDLE};
    void* mapped{nullptr};
    createMappedBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, buffer, allocation, mapped);
    if (storage.buffer != VK_NULL_HANDLE) {
        vmaUnmapMemory(graphics::internal::context.allocator, storage.allocation);
        vmaDestroyBuffer(graphics::internal::context.allocator, storage.buffer, storage.allocation);
    }
    storage.buffer = buffer;
    storage.allocation = allocation;
    storage.mapped = mapped;
    storage.capacity = capacity;
    updateStorageDescriptor(binding, storage, bufferSize);
}

void ensureStorageCapacity(MappedStorageBuffer& storage, size_t count,
                           VkDeviceSize elementSize, uint32_t binding) {
    if (count <= storage.capacity) {
        return;
    }
    VkDeviceSize newCapacity = std::max<VkDeviceSize>(storage.capacity, 1);
    while (newCapacity < count) {
        newCapacity *= 2;
    }
    checkVk(vkDeviceWaitIdle(device()), "vkDeviceWaitIdle");
    createStorageBuffer(storage, newCapacity, elementSize, binding);
}

void createUniformBuffers() {
    createMappedBuffer(sizeof(SceneUniformBuffer), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                       sceneUniformBuffer, sceneUniformBufferAllocation, sceneUniformBufferMapped);
    createMappedBuffer(sizeof(ModelUniformBuffer), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                       modelUniformBuffer, modelUniformBufferAllocation, modelUniformBufferMapped);
}

void createDescriptorPoolAndSets() {
    const std::array<VkDescriptorPoolSize, 2> poolSizes{{
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
    }};
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 2;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    checkVk(vkCreateDescriptorPool(device(), &poolInfo, nullptr, &descriptorPool),
            "vkCreateDescriptorPool");

    const std::array<VkDescriptorSetLayout, 2> layouts{
        sceneDescriptorSetLayout, modelDescriptorSetLayout};
    std::array<VkDescriptorSet, 2> sets{};
    VkDescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo.descriptorPool = descriptorPool;
    allocateInfo.descriptorSetCount = static_cast<uint32_t>(layouts.size());
    allocateInfo.pSetLayouts = layouts.data();
    checkVk(vkAllocateDescriptorSets(device(), &allocateInfo, sets.data()), "vkAllocateDescriptorSets");
    sceneDescriptorSet = sets[0];
    modelDescriptorSet = sets[1];

    const VkDeviceSize pointCapacity = std::max<size_t>(pointLights.size(), 1);
    const VkDeviceSize spotCapacity = std::max<size_t>(spotLights.size(), 1);
    createStorageBuffer(pointLightBuffer, pointCapacity, sizeof(PointLightGpu), 1);
    createStorageBuffer(spotLightBuffer, spotCapacity, sizeof(SpotLightGpu), 2);

    const VkDescriptorBufferInfo sceneBufferInfo{sceneUniformBuffer, 0, sizeof(SceneUniformBuffer)};
    const VkDescriptorBufferInfo modelBufferInfo{modelUniformBuffer, 0, sizeof(ModelUniformBuffer)};
    const std::array<VkWriteDescriptorSet, 2> writes{{
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sceneDescriptorSet, 0, 0, 1,
         VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &sceneBufferInfo, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, modelDescriptorSet, 0, 0, 1,
         VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &modelBufferInfo, nullptr},
    }};
    vkUpdateDescriptorSets(device(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void createPipeline() {
    const auto vertCode = readFile("shaders/tetrahedron.vert.spv");
    const auto fragCode = readFile("shaders/tetrahedron.frag.spv");
    const VkShaderModule vertModule = createShaderModule(vertCode);
    const VkShaderModule fragModule = createShaderModule(fragCode);

    const std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages{{
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
         VK_SHADER_STAGE_VERTEX_BIT, vertModule, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
         VK_SHADER_STAGE_FRAGMENT_BIT, fragModule, "main", nullptr},
    }};

    const VkVertexInputBindingDescription bindingDescription = Vertex::getBindingDescription();
    const auto attributeDescriptions = Vertex::getAttributeDescriptions();
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = nullptr;
    viewportState.scissorCount = 1;
    viewportState.pScissors = nullptr;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                     VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &blendAttachment;

    const std::array<VkDynamicState, 2> dynamicStates{
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    const std::array<VkDescriptorSetLayout, 2> setLayouts{
        sceneDescriptorSetLayout, modelDescriptorSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
    pipelineLayoutInfo.pSetLayouts = setLayouts.data();
    checkVk(vkCreatePipelineLayout(device(), &pipelineLayoutInfo, nullptr, &pipelineLayout),
            "vkCreatePipelineLayout");

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = graphics::internal::context.render_pass;
    pipelineInfo.subpass = 0;
    checkVk(vkCreateGraphicsPipelines(device(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline),
            "vkCreateGraphicsPipelines");

    vkDestroyShaderModule(device(), fragModule, nullptr);
    vkDestroyShaderModule(device(), vertModule, nullptr);
}

glm::vec3 cameraForward() {
    const float yaw = glm::radians(cameraYaw);
    const float pitch = glm::radians(cameraPitch);
    return glm::normalize(glm::vec3(std::cos(yaw) * std::cos(pitch),
                                    std::sin(pitch),
                                    std::sin(yaw) * std::cos(pitch)));
}

glm::vec3 safeNormalize(glm::vec3 value, glm::vec3 fallback) {
    const float lengthSquared = glm::dot(value, value);
    return lengthSquared > 1.0e-8f ? value / std::sqrt(lengthSquared) : fallback;
}

void updateUniformBuffers() {
    const auto& extent = graphics::internal::context.swapchain_extent;
    if (extent.width == 0 || extent.height == 0) {
        return;
    }
    ensureStorageCapacity(pointLightBuffer, pointLights.size(), sizeof(PointLightGpu), 1);
    ensureStorageCapacity(spotLightBuffer, spotLights.size(), sizeof(SpotLightGpu), 2);

    const glm::mat4 model = glm::translate(glm::mat4(1.0f), modelPosition) *
        glm::rotate(glm::mat4(1.0f), glm::radians(modelRotation.x), {1.0f, 0.0f, 0.0f}) *
        glm::rotate(glm::mat4(1.0f), glm::radians(modelRotation.y), {0.0f, 1.0f, 0.0f}) *
        glm::rotate(glm::mat4(1.0f), glm::radians(modelRotation.z), {0.0f, 0.0f, 1.0f}) *
        glm::scale(glm::mat4(1.0f), glm::vec3(modelScale));
    const glm::vec3 forward = cameraForward();
    const glm::mat4 view = glm::lookAt(cameraPosition, cameraPosition + forward,
                                       {0.0f, 1.0f, 0.0f});
    const float aspectRatio = static_cast<float>(extent.width) / static_cast<float>(extent.height);
    glm::mat4 projection = usePerspective
        ? glm::perspective(glm::radians(fov), aspectRatio, 0.1f, 100.0f)
        : glm::orthoRH_ZO(-orthographicSize * aspectRatio, orthographicSize * aspectRatio,
                          -orthographicSize, orthographicSize, 0.1f, 100.0f);
    projection[1][1] *= -1.0f;

    const SceneUniformBuffer scene{
        projection * view,
        glm::vec4(cameraPosition, 1.0f),
        glm::vec4(ambientColor, ambientIntensity),
        glm::vec4(safeNormalize(directionalDirection, {0.0f, -1.0f, 0.0f}), 0.0f),
        glm::vec4(directionalColor, directionalIntensity),
        glm::uvec4(static_cast<uint32_t>(pointLights.size()),
                   static_cast<uint32_t>(spotLights.size()), 0, 0),
    };
    const ModelUniformBuffer modelData{
        model,
        albedo,
        glm::vec4(specularColor, 1.0f),
        glm::vec4(shininess, 0.0f, 0.0f, 0.0f),
    };
    std::memcpy(sceneUniformBufferMapped, &scene, sizeof(scene));
    std::memcpy(modelUniformBufferMapped, &modelData, sizeof(modelData));
    checkVk(vmaFlushAllocation(graphics::internal::context.allocator,
                               sceneUniformBufferAllocation, 0, sizeof(scene)),
            "vmaFlushAllocation(scene)");
    checkVk(vmaFlushAllocation(graphics::internal::context.allocator,
                               modelUniformBufferAllocation, 0, sizeof(modelData)),
            "vmaFlushAllocation(model)");
    auto* pointData = static_cast<PointLightGpu*>(pointLightBuffer.mapped);
    for (size_t i = 0; i < pointLights.size(); ++i) {
        pointData[i] = {glm::vec4(pointLights[i].position, 1.0f),
                        glm::vec4(pointLights[i].color, pointLights[i].intensity)};
    }
    auto* spotData = static_cast<SpotLightGpu*>(spotLightBuffer.mapped);
    for (size_t i = 0; i < spotLights.size(); ++i) {
        const auto& light = spotLights[i];
        const float inner = glm::radians(std::min(light.innerAngle, light.outerAngle));
        const float outer = glm::radians(std::max(light.innerAngle, light.outerAngle));
        spotData[i] = {
            glm::vec4(light.position, 1.0f),
            glm::vec4(safeNormalize(light.direction, {0.0f, -1.0f, 0.0f}), 0.0f),
            glm::vec4(light.color, light.intensity),
            glm::vec4(std::cos(inner), std::cos(outer), 0.0f, 0.0f),
        };
    }
    checkVk(vmaFlushAllocation(graphics::internal::context.allocator, pointLightBuffer.allocation,
                               0, pointLightBuffer.capacity * sizeof(PointLightGpu)),
            "vmaFlushAllocation(point lights)");
    checkVk(vmaFlushAllocation(graphics::internal::context.allocator, spotLightBuffer.allocation,
                               0, spotLightBuffer.capacity * sizeof(SpotLightGpu)),
            "vmaFlushAllocation(spotlights)");
}

} // namespace

VkPipeline pipeline{VK_NULL_HANDLE};
VkPipelineLayout pipelineLayout{VK_NULL_HANDLE};
VkBuffer vertexBuffer{VK_NULL_HANDLE};
VkBuffer indexBuffer{VK_NULL_HANDLE};
VmaAllocation vertexBufferAllocation{VK_NULL_HANDLE};
VmaAllocation indexBufferAllocation{VK_NULL_HANDLE};
std::vector<Vertex> vertices;
std::vector<uint32_t> indices;

VkVertexInputBindingDescription Vertex::getBindingDescription() {
    return {0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
}

std::array<VkVertexInputAttributeDescription, 2> Vertex::getAttributeDescriptions() {
    return {{
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
    }};
}

bool initialize() {
    try {
        createTruncatedTetrahedron();
        createGeometryBuffers();
        createDescriptorSetLayouts();
        createUniformBuffers();
        createDescriptorPoolAndSets();
        createPipeline();
        return true;
    } catch (const std::exception& error) {
        std::cerr << "Initialization error: " << error.what() << '\n';
        shutdown();
        return false;
    }
}

void update(double time) {
    static double lastTime = time;
    const double deltaTime = time - lastTime;
    lastTime = time;
    if (isAnimating) {
        animationTime += static_cast<float>(deltaTime) * animationSpeed;
        modelPosition = {
            trajectoryRadius * std::sin(animationTime),
            std::sin(animationTime * 1.5f) * 0.5f,
            trajectoryRadius * std::cos(animationTime * 0.5f),
        };
        modelRotation = {
            std::sin(animationTime) * 15.0f,
            animationTime * 30.0f,
            0.0f,
        };
    }
    buildImGui();
    updateUniformBuffers();
}

void render(const graphics::internal::FrameData& frame) {
    if (frame.command_buffer == VK_NULL_HANDLE || pipeline == VK_NULL_HANDLE) {
        return;
    }

    const VkCommandBuffer commandBuffer = frame.command_buffer;
    const auto& context = graphics::internal::context;
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    checkVk(vkBeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer");

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = context.render_pass;
    renderPassInfo.framebuffer = frame.framebuffer;
    renderPassInfo.renderArea.extent = context.swapchain_extent;
    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = {{0.025f, 0.03f, 0.045f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};
    renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();
    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport{0.0f, 0.0f,
                        static_cast<float>(context.swapchain_extent.width),
                        static_cast<float>(context.swapchain_extent.height), 0.0f, 1.0f};
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    VkRect2D scissor{{0, 0}, context.swapchain_extent};
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
    const VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, &offset);
    vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    const std::array<VkDescriptorSet, 2> sets{sceneDescriptorSet, modelDescriptorSet};
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                            0, static_cast<uint32_t>(sets.size()), sets.data(), 0, nullptr);
    vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
    vkCmdEndRenderPass(commandBuffer);
    checkVk(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");
}

void buildImGui() {
    ImGui::Begin("Blinn-Phong lighting | Truncated tetrahedron");
    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::DragFloat3("Position##camera", &cameraPosition.x, 0.05f, -20.0f, 20.0f);
        ImGui::SliderFloat("Yaw", &cameraYaw, -180.0f, 180.0f);
        ImGui::SliderFloat("Pitch", &cameraPitch, -89.0f, 89.0f);
        ImGui::Checkbox("Perspective projection", &usePerspective);
        if (usePerspective) {
            ImGui::SliderFloat("Field of view", &fov, 15.0f, 100.0f, "%.1f deg");
        } else {
            ImGui::SliderFloat("Orthographic size", &orthographicSize, 1.0f, 10.0f);
        }
    }
    if (ImGui::CollapsingHeader("Model and material", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::DragFloat3("Position##model", &modelPosition.x, 0.02f, -5.0f, 5.0f);
        ImGui::SliderFloat3("Rotation (degrees)", &modelRotation.x, -180.0f, 180.0f);
        ImGui::SliderFloat("Scale", &modelScale, 0.1f, 3.0f);
        ImGui::ColorEdit3("Albedo", &albedo.x);
        ImGui::ColorEdit3("Specular color", &specularColor.x);
        ImGui::SliderFloat("Shininess", &shininess, 1.0f, 256.0f, "%.0f");
        ImGui::Checkbox("Animate model", &isAnimating);
        if (isAnimating) {
            ImGui::SliderFloat("Animation speed", &animationSpeed, 0.1f, 5.0f);
            ImGui::SliderFloat("Trajectory radius", &trajectoryRadius, 0.5f, 5.0f);
        }
    }
    if (ImGui::CollapsingHeader("Ambient light", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::ColorEdit3("Ambient color", &ambientColor.x);
        ImGui::SliderFloat("Ambient intensity", &ambientIntensity, 0.0f, 2.0f);
    }
    if (ImGui::CollapsingHeader("Directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::DragFloat3("Direction##directional", &directionalDirection.x, 0.02f, -1.0f, 1.0f);
        ImGui::ColorEdit3("Color##directional", &directionalColor.x);
        ImGui::SliderFloat("Intensity##directional", &directionalIntensity, 0.0f, 5.0f);
    }
    if (ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::Button("Add point light")) {
            pointLights.emplace_back();
        }
        for (size_t i = 0; i < pointLights.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            ImGui::Separator();
            ImGui::Text("Point light %zu", i + 1);
            ImGui::DragFloat3("Position", &pointLights[i].position.x, 0.05f, -10.0f, 10.0f);
            ImGui::ColorEdit3("Color", &pointLights[i].color.x);
            ImGui::SliderFloat("Intensity", &pointLights[i].intensity, 0.0f, 100.0f);
            if (ImGui::Button("Remove point light")) {
                pointLights.erase(pointLights.begin() + static_cast<std::ptrdiff_t>(i));
                ImGui::PopID();
                break;
            }
            ImGui::PopID();
        }
    }
    if (ImGui::CollapsingHeader("Spotlights", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::Button("Add spotlight")) {
            spotLights.emplace_back();
        }
        for (size_t i = 0; i < spotLights.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            ImGui::Separator();
            ImGui::Text("Spotlight %zu", i + 1);
            ImGui::DragFloat3("Position", &spotLights[i].position.x, 0.05f, -10.0f, 10.0f);
            ImGui::DragFloat3("Direction", &spotLights[i].direction.x, 0.02f, -1.0f, 1.0f);
            ImGui::ColorEdit3("Color", &spotLights[i].color.x);
            ImGui::SliderFloat("Intensity", &spotLights[i].intensity, 0.0f, 100.0f);
            ImGui::SliderFloat("Inner angle", &spotLights[i].innerAngle, 1.0f, 88.5f, "%.1f deg");
            ImGui::SliderFloat("Outer angle", &spotLights[i].outerAngle, 1.0f, 89.0f, "%.1f deg");
            if (spotLights[i].outerAngle < spotLights[i].innerAngle + 0.5f) {
                spotLights[i].outerAngle = std::min(spotLights[i].innerAngle + 0.5f, 89.0f);
            }
            if (ImGui::Button("Remove spotlight")) {
                spotLights.erase(spotLights.begin() + static_cast<std::ptrdiff_t>(i));
                ImGui::PopID();
                break;
            }
            ImGui::PopID();
        }
    }
    ImGui::Separator();
    ImGui::Text("Vertices: %zu | Triangles: %zu", vertices.size(), indices.size() / 3);
    ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);
    ImGui::End();
}

void shutdown() {
    if (device() == VK_NULL_HANDLE) {
        return;
    }
    vkDeviceWaitIdle(device());
    vkDestroyPipeline(device(), pipeline, nullptr);
    vkDestroyPipelineLayout(device(), pipelineLayout, nullptr);
    vkDestroyDescriptorPool(device(), descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device(), modelDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(device(), sceneDescriptorSetLayout, nullptr);
    if (vertexBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(graphics::internal::context.allocator, vertexBuffer, vertexBufferAllocation);
    }
    if (indexBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(graphics::internal::context.allocator, indexBuffer, indexBufferAllocation);
    }
    if (sceneUniformBuffer != VK_NULL_HANDLE) {
        if (sceneUniformBufferMapped != nullptr) {
            vmaUnmapMemory(graphics::internal::context.allocator, sceneUniformBufferAllocation);
        }
        vmaDestroyBuffer(graphics::internal::context.allocator, sceneUniformBuffer, sceneUniformBufferAllocation);
    }
    if (modelUniformBuffer != VK_NULL_HANDLE) {
        if (modelUniformBufferMapped != nullptr) {
            vmaUnmapMemory(graphics::internal::context.allocator, modelUniformBufferAllocation);
        }
        vmaDestroyBuffer(graphics::internal::context.allocator, modelUniformBuffer, modelUniformBufferAllocation);
    }
    for (auto* storage : {&pointLightBuffer, &spotLightBuffer}) {
        if (storage->buffer != VK_NULL_HANDLE) {
            vmaUnmapMemory(graphics::internal::context.allocator, storage->allocation);
            vmaDestroyBuffer(graphics::internal::context.allocator, storage->buffer, storage->allocation);
        }
    }
}

} // namespace application
