#include "rt_material.cuh"
#include "toml11/toml.hpp"
#include "rt_raytracing.cuh"
#include <map>
#include <string>
#include <iostream>
#include <utility>
#include <fmt/core.h>
#include "renderer/device_texture.cuh"

class MaterialDefinition {
public:
    MaterialDefinition()
            : emission_color({0, 0, 0}), emission_intensity(0.0f), reflectivity(0.0f) {
    }

    glm::vec3 emission_color;
    float emission_intensity;
    float reflectivity;
};

class MaterialsContext {
public:
    MaterialsContext() = default;

    explicit MaterialsContext(std::map<std::string, MaterialDefinition, std::less<>> defs) : m_material_definitions(
            std::move(defs)) {}

    MaterialDefinition get_or_default(std::string_view name) {
        auto it = m_material_definitions.find(name);
        if (it == m_material_definitions.end()) {
            MaterialDefinition default_material;
            m_material_definitions.insert({std::string(name), default_material});

            return default_material;
        }

        return it->second;
    }

private:
    std::map<std::string, MaterialDefinition, std::less<>> m_material_definitions;
};

MaterialsContext materials_context;

void RT_InitMaterials(const RayTracingInitOptions &options) {
    std::map<std::string, MaterialDefinition, std::less<>> material_definitions;
    if (options.materials_file.has_value()) {
        std::cout << fmt::format("Loading materials file: {}\n", options.materials_file.value().c_str());
        const auto data = toml::parse(options.materials_file.value());
        auto materials = toml::find<std::vector<toml::value>>(data, "materials");
        for (auto &mat: materials) {
            std::vector<std::string> texture_names;
            if (mat.contains("texture")) {
                texture_names.push_back(toml::find<std::string>(mat, "texture"));
            }

            if (mat.contains("textures")) {
                texture_names = toml::find<std::vector<std::string>>(mat, "textures");
            }

            if(mat.contains("sprite_group")) {
                auto texture_group = toml::find<std::string>(mat, "sprite_group");
                for(auto &sprite_name : detail::sprite_data->sprite_names()){
                    if (sprite_name.rfind(texture_group, 0) != 0) {
                        continue;
                    }

                    texture_names.push_back(sprite_name);
                }
            }

            if (texture_names.empty()) {
                continue;
            }

            for (auto &texture_name: texture_names) {
                if (material_definitions.count(texture_name) == 1) {
                    std::cerr << fmt::format("Found duplicate material definitions for {}\n", texture_name);
                    continue;
                }

                MaterialDefinition material_definition;


                material_definition.emission_intensity = toml::find_or<float>(mat, "emission_intensity", 0.0f);

                auto emission = toml::find_or<std::vector<float>>(mat, "emission", {0, 0, 0});
                material_definition.emission_color = glm::vec3(emission[0], emission[1], emission[2]);

                material_definition.reflectivity = toml::find_or<float>(mat, "reflectivity", 0.0f);

                material_definitions[texture_name] = material_definition;
            }

        }
    }

    materials_context = MaterialsContext(material_definitions);
}

DeviceMaterial RT_GetMaterial(std::string_view name, DeviceTexture *texture) {
    DeviceMaterial material(texture);
    auto material_definition = materials_context.get_or_default(name);

    if (material_definition.emission_intensity > 0.0f && material_definition.emission_color.x == 0.0f &&
        material_definition.emission_color.y == 0.0f && material_definition.emission_color.z == 0.0f) {
        glm::vec3 color_sum{};
        int pixel_count = 0;
        for (int i = 0; i < texture->width() * texture->height(); ++i) {
            auto palette_index = texture->data()[i];
            if (palette_index > 0xFF) {
                continue;
            }

            auto color = glm::vec3{
                    device::palette[palette_index * 3] / 255.0f,
                    device::palette[(palette_index * 3) + 1] / 255.0f,
                    device::palette[(palette_index * 3) + 2] / 255.0f,
            };

            color_sum += color;
            pixel_count++;
        }

        material_definition.emission_color = color_sum / static_cast<float>(pixel_count);
    }

    material.set_emission(material_definition.emission_color * material_definition.emission_intensity);
    material.set_reflectivity(material_definition.reflectivity);


    return material;
}
