#version 450

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragUv;
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

layout(set = 2, binding = 0) uniform sampler2D albedoTexture;
layout(set = 2, binding = 1) uniform sampler2D specularTexture;
layout(set = 2, binding = 2) uniform sampler2D emissiveTexture;

layout(push_constant) uniform SamplingParameters {
    uint enabled;
    float time;
    float intensity;
    float uvScale;
} sampling;

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

vec3 blinnPhong(vec3 normal, vec3 viewDirection, vec3 lightDirection, vec3 radiance,
                vec3 albedo, float specularMask) {
    float diffuseFactor = max(dot(normal, lightDirection), 0.0);
    float specularFactor = 0.0;
    if (diffuseFactor > 0.0) {
        vec3 halfwayDirection = normalize(lightDirection + viewDirection);
        specularFactor = pow(max(dot(normal, halfwayDirection), 0.0),
                             max(material.materialParameters.x, 1.0));
    }
    return radiance * (albedo * diffuseFactor +
                       material.specularColor.rgb * specularMask * specularFactor);
}

void main() {
    vec3 surfaceColor;
    if (sampling.enabled != 0u) {
        vec2 movingUv = fragUv * sampling.uvScale +
                        vec2(sin(sampling.time + fragUv.y * 8.0),
                             cos(sampling.time * 0.7 + fragUv.x * 7.0)) * 0.08;
        vec2 secondUv = fragUv * (sampling.uvScale * 1.37) -
                        vec2(sampling.time * 0.07, -sampling.time * 0.11);
        vec3 firstSample = texture(albedoTexture, movingUv).rgb;
        vec3 secondSample = texture(albedoTexture, secondUv).rgb;
        float modulation = 0.78 + 0.22 * sin(sampling.time * 2.0 + fragUv.x * 12.0);
        surfaceColor = mix(firstSample, secondSample, 0.42) *
                       modulation * sampling.intensity * material.albedo.rgb;
    } else {
        surfaceColor = texture(albedoTexture, fragUv).rgb * material.albedo.rgb;
    }

    vec3 normal = normalize(fragNormal);
    vec3 viewDirection = normalize(scene.cameraPosition.xyz - fragPosition);
    float specularMask = texture(specularTexture, fragUv).r;
    vec3 lighting = surfaceColor * scene.ambientColorIntensity.rgb *
                    scene.ambientColorIntensity.a;

    vec3 directional = normalize(-scene.directionalDirection.xyz);
    lighting += blinnPhong(normal, viewDirection, directional,
                           scene.directionalColorIntensity.rgb *
                           scene.directionalColorIntensity.a, surfaceColor, specularMask);

    for (uint i = 0; i < scene.lightCounts.x; ++i) {
        PointLight light = pointLightBuffer.lights[i];
        vec3 toLight = light.position.xyz - fragPosition;
        float distanceSquared = max(dot(toLight, toLight), 0.0001);
        vec3 lightDirection = toLight * inversesqrt(distanceSquared);
        vec3 radiance = light.colorIntensity.rgb *
                        (light.colorIntensity.a / distanceSquared);
        lighting += blinnPhong(normal, viewDirection, lightDirection, radiance,
                               surfaceColor, specularMask);
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
        lighting += blinnPhong(normal, viewDirection, lightDirection, radiance,
                               surfaceColor, specularMask);
    }

    lighting += texture(emissiveTexture, fragUv).rgb;
    outColor = vec4(lighting, material.albedo.a);
}
