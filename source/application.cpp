#include "application.hpp"
#include "graphics_internal.hpp"

#include <imgui.h>
#include <array>
#include <vk_mem_alloc.h>
#include <fstream>
#include <stdexcept>
#include <iostream>

namespace application {

void buildImGui();

// 1. GLOBAL VARIABLES
std::vector<Vertex> vertices;
std::vector<uint32_t> indices;

VkPipeline pipeline = VK_NULL_HANDLE;
VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

VkBuffer vertexBuffer = VK_NULL_HANDLE;
VmaAllocation vertexBufferAllocation = VK_NULL_HANDLE;

VkBuffer indexBuffer = VK_NULL_HANDLE;
VmaAllocation indexBufferAllocation = VK_NULL_HANDLE;

VkBuffer uniformBuffer = VK_NULL_HANDLE;
VmaAllocation uniformBufferAllocation = VK_NULL_HANDLE;
void* uniformBufferMapped = nullptr;

float rotationSpeed = 0.5f;
float scale = 1.0f;
bool usePerspective = true;
float fov = 45.0f;
glm::vec3 rotation = {0.0f, 0.0f, 0.0f};

// 2. HELPER FUNCTIONS

static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

static VkShaderModule createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(graphics::internal::context.device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module!");
    }
    return shaderModule;
}

VkVertexInputBindingDescription Vertex::getBindingDescription() {
    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return bindingDescription;
}

std::array<VkVertexInputAttributeDescription, 2> Vertex::getAttributeDescriptions() {
    std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};
    
    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(Vertex, pos);

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, color);

    return attributeDescriptions;
}

static void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage, VkBuffer& buffer, VmaAllocation& allocation) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = memoryUsage;

    auto& ctx = graphics::internal::context;
    if (vmaCreateBuffer(ctx.allocator, &bufferInfo, &allocInfo, &buffer, &allocation, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer!");
    }
}

// 3. GEOMETRY: TRUNCATED TETRAHEDRON
void createTruncatedTetrahedron() {
    glm::vec3 v0 = { 1.0f,  1.0f,  1.0f};
    glm::vec3 v1 = { 1.0f, -1.0f, -1.0f};
    glm::vec3 v2 = {-1.0f,  1.0f, -1.0f};
    glm::vec3 v3 = {-1.0f, -1.0f,  1.0f};

    auto lerp = [](glm::vec3 a, glm::vec3 b, float t) {
        return a + (b - a) * t;
    };

    glm::vec3 e01_a = lerp(v0, v1, 1.0f/3.0f), e01_b = lerp(v0, v1, 2.0f/3.0f);
    glm::vec3 e02_a = lerp(v0, v2, 1.0f/3.0f), e02_b = lerp(v0, v2, 2.0f/3.0f);
    glm::vec3 e03_a = lerp(v0, v3, 1.0f/3.0f), e03_b = lerp(v0, v3, 2.0f/3.0f);
    glm::vec3 e12_a = lerp(v1, v2, 1.0f/3.0f), e12_b = lerp(v1, v2, 2.0f/3.0f);
    glm::vec3 e13_a = lerp(v1, v3, 1.0f/3.0f), e13_b = lerp(v1, v3, 2.0f/3.0f);
    glm::vec3 e23_a = lerp(v2, v3, 1.0f/3.0f), e23_b = lerp(v2, v3, 2.0f/3.0f);

    auto addFace = [&](const std::vector<glm::vec3>& faceVerts, glm::vec3 color) {
        uint32_t startIndex = static_cast<uint32_t>(vertices.size());
        for (const auto& v : faceVerts) {
            vertices.push_back({v, color});
        }
        for (size_t i = 1; i < faceVerts.size() - 1; ++i) {
            indices.push_back(startIndex);
            indices.push_back(startIndex + static_cast<uint32_t>(i));
            indices.push_back(startIndex + static_cast<uint32_t>(i) + 1);
        }
    };

    addFace({e01_a, e02_a, e03_a}, {1.0f, 0.0f, 0.0f});
    addFace({e01_b, e13_a, e12_a}, {0.0f, 1.0f, 0.0f});
    addFace({e02_b, e12_b, e23_a}, {0.0f, 0.0f, 1.0f});
    addFace({e03_b, e13_b, e23_b}, {1.0f, 1.0f, 0.0f});

    addFace({e01_a, e01_b, e12_a, e12_b, e02_b, e02_a}, {0.0f, 1.0f, 1.0f});
    addFace({e01_a, e03_a, e03_b, e13_b, e13_a, e01_b}, {1.0f, 0.0f, 1.0f});
    addFace({e02_a, e02_b, e23_a, e23_b, e03_b, e03_a}, {1.0f, 0.5f, 0.0f});
    addFace({e12_a, e12_b, e23_a, e23_b, e13_b, e13_a}, {1.0f, 1.0f, 1.0f});
}

// 4. VULKAN INITIALIZATION
void createDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &uboLayoutBinding;

    auto& ctx = graphics::internal::context;
    vkCreateDescriptorSetLayout(ctx.device, &layoutInfo, nullptr, &descriptorSetLayout);
}

void createBuffers() {
    auto& ctx = graphics::internal::context;
    VkDeviceSize vertexSize = sizeof(vertices[0]) * vertices.size();
    VkDeviceSize indexSize = sizeof(indices[0]) * indices.size();

    VkBuffer stagingVertexBuffer, stagingIndexBuffer;
    VmaAllocation stagingVertexAlloc, stagingIndexAlloc;

    createBuffer(vertexSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY, stagingVertexBuffer, stagingVertexAlloc);
    createBuffer(indexSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY, stagingIndexBuffer, stagingIndexAlloc);

    void* data;
    vmaMapMemory(ctx.allocator, stagingVertexAlloc, &data);
    memcpy(data, vertices.data(), vertexSize);
    vmaUnmapMemory(ctx.allocator, stagingVertexAlloc);

    vmaMapMemory(ctx.allocator, stagingIndexAlloc, &data);
    memcpy(data, indices.data(), indexSize);
    vmaUnmapMemory(ctx.allocator, stagingIndexAlloc);

    createBuffer(vertexSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, vertexBuffer, vertexBufferAllocation);
    createBuffer(indexSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, indexBuffer, indexBufferAllocation);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolInfo.queueFamilyIndex = ctx.graphics_queue_index;
    
    VkCommandPool tempCommandPool;
    vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &tempCommandPool);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = tempCommandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(ctx.device, &allocInfo, &cmd);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.size = vertexSize;
    vkCmdCopyBuffer(cmd, stagingVertexBuffer, vertexBuffer, 1, &copyRegion);
    copyRegion.size = indexSize;
    vkCmdCopyBuffer(cmd, stagingIndexBuffer, indexBuffer, 1, &copyRegion);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    vkQueueSubmit(ctx.graphics_queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx.graphics_queue);

    vmaDestroyBuffer(ctx.allocator, stagingVertexBuffer, stagingVertexAlloc);
    vmaDestroyBuffer(ctx.allocator, stagingIndexBuffer, stagingIndexAlloc);
    vkFreeCommandBuffers(ctx.device, tempCommandPool, 1, &cmd);
    vkDestroyCommandPool(ctx.device, tempCommandPool, nullptr);
}

void createUniformBuffer() {
    VkDeviceSize bufferSize = sizeof(UniformBufferObject);
    createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, uniformBuffer, uniformBufferAllocation);
    vmaMapMemory(graphics::internal::context.allocator, uniformBufferAllocation, &uniformBufferMapped);
}

void createDescriptorPoolAndSet() {
    auto& ctx = graphics::internal::context;
    
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize.descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 1;
    vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &descriptorPool);

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;
    vkAllocateDescriptorSets(ctx.device, &allocInfo, &descriptorSet);

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = uniformBuffer;
    bufferInfo.offset = 0;
    bufferInfo.range = sizeof(UniformBufferObject);

    VkWriteDescriptorSet descriptorWrite{};
    descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite.dstSet = descriptorSet;
    descriptorWrite.dstBinding = 0;
    descriptorWrite.dstArrayElement = 0;
    descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(ctx.device, 1, &descriptorWrite, 0, nullptr);
}

void createPipeline() {
    auto& ctx = graphics::internal::context;
    auto vertShaderCode = readFile("shaders/tetrahedron.vert.spv");
    auto fragShaderCode = readFile("shaders/tetrahedron.frag.spv");

    VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)ctx.swapchain_extent.width;
    viewport.height = (float)ctx.swapchain_extent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = ctx.swapchain_extent;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

    vkCreatePipelineLayout(ctx.device, &pipelineLayoutInfo, nullptr, &pipelineLayout);

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = ctx.render_pass;
    pipelineInfo.subpass = 0;

    vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);

    vkDestroyShaderModule(ctx.device, fragShaderModule, nullptr);
    vkDestroyShaderModule(ctx.device, vertShaderModule, nullptr);
}

bool initialize() {
    try {
        createTruncatedTetrahedron();
        createDescriptorSetLayout();
        createBuffers();
        createUniformBuffer();
        createDescriptorPoolAndSet();
        createPipeline();
        return true;
    } 
    catch (const std::exception& e) {
        std::cerr << "Initialization error: " << e.what() << std::endl;
        return false;
    }
}

// 5. DATA UPDATE
void updateUniformBuffer() {
    UniformBufferObject ubo{};
    
    auto& ctx = graphics::internal::context;
    if (ctx.swapchain_extent.width == 0 || ctx.swapchain_extent.height == 0) {
        return;
    }

    ubo.model = glm::mat4(1.0f);
    ubo.model = glm::scale(ubo.model, glm::vec3(scale, scale, scale));
    ubo.model = glm::rotate(ubo.model, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
    ubo.model = glm::rotate(ubo.model, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
    ubo.model = glm::rotate(ubo.model, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

    ubo.view = glm::lookAt(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    float aspectRatio = (float)ctx.swapchain_extent.width / (float)ctx.swapchain_extent.height;
    if (usePerspective) {
        ubo.proj = glm::perspective(glm::radians(fov), aspectRatio, 0.1f, 100.0f);
    } else {
        float orthoSize = 3.0f;
        ubo.proj = glm::ortho(-orthoSize * aspectRatio, orthoSize * aspectRatio, -orthoSize, orthoSize, 0.1f, 100.0f);
    }
    
    ubo.proj[1][1] *= -1.0f;

    memcpy(uniformBufferMapped, &ubo, sizeof(ubo));
}

void update(double time) {
    static double lastTime = time;
    double deltaTime = time - lastTime;
    lastTime = time;
    
    rotation.y += rotationSpeed * (float)deltaTime * 10.0f;
    
    updateUniformBuffer();
    buildImGui();
}

// 6. RENDERING
void render(const graphics::internal::FrameData& fd) {
    if (fd.command_buffer == VK_NULL_HANDLE || pipeline == VK_NULL_HANDLE) {
        return;
    }

    VkCommandBuffer cmd = fd.command_buffer;
    auto& ctx = graphics::internal::context;

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = ctx.render_pass;
    renderPassInfo.framebuffer = fd.framebuffer;
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = ctx.swapchain_extent;

    VkClearValue clearValues[2];
    clearValues[0].color = {{0.1f, 0.1f, 0.1f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};
    
    renderPassInfo.clearValueCount = 2;
    renderPassInfo.pClearValues = clearValues;

    vkCmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)ctx.swapchain_extent.width;
    viewport.height = (float)ctx.swapchain_extent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = ctx.swapchain_extent;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    VkBuffer vertexBuffers[] = {vertexBuffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

    vkCmdDrawIndexed(cmd, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
}

// 7. IMGUI INTERFACE
void buildImGui() {
    static bool fontsLoaded = false;
    if (!fontsLoaded) {
        ImGuiIO& io = ImGui::GetIO();
        io.Fonts->AddFontDefault();
        fontsLoaded = true;
    }
    
    ImGui::Begin("Truncated Tetrahedron (Variant 4)");
    
    ImGui::Text("Transformation and Projection Controls");
    ImGui::Separator();
    
    ImGui::SliderFloat("Rotation Speed", &rotationSpeed, 0.0f, 5.0f);
    ImGui::SliderFloat("Scale", &scale, 0.1f, 3.0f);
    
    ImGui::Checkbox("Perspective Projection", &usePerspective);
    if (usePerspective) {
        ImGui::SliderFloat("Field of View (FOV)", &fov, 10.0f, 120.0f);
    }
    
    ImGui::Separator();
    ImGui::Text("Vertices: %zu, Indices: %zu", vertices.size(), indices.size());
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    
    ImGui::End();
}

// 8. CLEANUP
void shutdown() {
    auto& ctx = graphics::internal::context;
    
    vkDeviceWaitIdle(ctx.device);

    vkDestroyPipeline(ctx.device, pipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, descriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(ctx.device, descriptorPool, nullptr);

    vmaDestroyBuffer(ctx.allocator, vertexBuffer, vertexBufferAllocation);
    vmaDestroyBuffer(ctx.allocator, indexBuffer, indexBufferAllocation);
    
    vmaUnmapMemory(ctx.allocator, uniformBufferAllocation);
    vmaDestroyBuffer(ctx.allocator, uniformBuffer, uniformBufferAllocation);
}

} // namespace application