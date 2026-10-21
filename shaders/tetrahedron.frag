#version 450

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0, std140) uniform SceneUniformBuffer {
    mat4 viewProjection;
    vec4 cameraPosition;
    vec4 ambientColorIntensity;
    vec4 directionalDirection;
    vec4 directionalColorIntensity;
    uvec4 lightCounts;
} scene;

layout(set = 1, binding = 0, std140) uniform ModelUniformBuffer {
    mat4 model;
    vec4 albedo;
    vec4 specularColor;
    vec4 materialParameters;
} material;

struct PointLight {
    vec4 position;
    vec4 colorIntensity;
};

struct SpotLight {
    vec4 position;
    vec4 direction;
    vec4 colorIntensity;
    vec4 coneCosines;
};

layout(set = 0, binding = 1, std430) readonly buffer PointLightBuffer {
    PointLight lights[];
} pointLightBuffer;

layout(set = 0, binding = 2, std430) readonly buffer SpotLightBuffer {
    SpotLight lights[];
} spotLightBuffer;

vec3 blinnPhong(vec3 normal, vec3 viewDirection, vec3 lightDirection, vec3 radiance) {
    float diffuseFactor = max(dot(normal, lightDirection), 0.0);
    float specularFactor = 0.0;
    if (diffuseFactor > 0.0) {
        vec3 halfwayDirection = normalize(lightDirection + viewDirection);
        specularFactor = pow(max(dot(normal, halfwayDirection), 0.0),
                             max(material.materialParameters.x, 1.0));
    }
    return radiance * (material.albedo.rgb * diffuseFactor +
                       material.specularColor.rgb * specularFactor);
}

void main() {
    vec3 normal = normalize(fragNormal);
    vec3 viewDirection = normalize(scene.cameraPosition.xyz - fragPosition);
    vec3 lighting = material.albedo.rgb * scene.ambientColorIntensity.rgb *
                    scene.ambientColorIntensity.a;

    vec3 directional = normalize(-scene.directionalDirection.xyz);
    lighting += blinnPhong(normal, viewDirection, directional,
                           scene.directionalColorIntensity.rgb *
                           scene.directionalColorIntensity.a);

    for (uint i = 0; i < scene.lightCounts.x; ++i) {
        PointLight light = pointLightBuffer.lights[i];
        vec3 toLight = light.position.xyz - fragPosition;
        float distanceSquared = max(dot(toLight, toLight), 0.0001);
        vec3 lightDirection = toLight * inversesqrt(distanceSquared);
        vec3 radiance = light.colorIntensity.rgb *
                        (light.colorIntensity.a / distanceSquared);
        lighting += blinnPhong(normal, viewDirection, lightDirection, radiance);
    }

    for (uint i = 0; i < scene.lightCounts.y; ++i) {
        SpotLight light = spotLightBuffer.lights[i];
        vec3 toLight = light.position.xyz - fragPosition;
        float distanceSquared = max(dot(toLight, toLight), 0.0001);
        vec3 lightDirection = toLight * inversesqrt(distanceSquared);
        vec3 fromLight = -lightDirection;
        float coneCosine = dot(fromLight, normalize(light.direction.xyz));
        float coneFactor = smoothstep(light.coneCosines.y, light.coneCosines.x, coneCosine);
        vec3 radiance = light.colorIntensity.rgb *
                        (light.colorIntensity.a * coneFactor / distanceSquared);
        lighting += blinnPhong(normal, viewDirection, lightDirection, radiance);
    }

    outColor = vec4(lighting, material.albedo.a);
}
