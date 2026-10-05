#version 450

layout(location = 0) in vec3 fragColor;

layout(std140, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 baseColor;
} ubo;

layout(location = 0) out vec4 outColor;

void main() {
    // Задание 5: Умножаем процедурный цвет вершины на цвет из интерфейса
    outColor = vec4(fragColor * ubo.baseColor.rgb, ubo.baseColor.a);
}