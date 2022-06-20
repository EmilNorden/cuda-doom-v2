#include "device_material.cuh"
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include "device_material_loader.cuh"


namespace fs = std::filesystem;
using namespace std;

vector <fs::path> get_image_files(std::string_view directory) {
    vector <fs::path> files;
    for (auto &file : fs::directory_iterator(directory)) {
        if (file.is_regular_file() && DeviceTextureLoader::file_is_supported(file.path())) {
            files.push_back(file.path());
        }
    }

    return files;
}

template<typename T>
void filter_files_by_filename(const vector<fs::path>& files, vector<fs::path> &out_files, T word) {
    string uppercase_word = word;
    std::transform(uppercase_word.begin(), uppercase_word.end(), uppercase_word.begin(),
                   [](unsigned char c) { return std::toupper(c); });

    for (auto &p : files) {
        string filename = p.filename();
        std::transform(filename.begin(), filename.end(), filename.begin(),
                       [](unsigned char c) { return std::toupper(c); });

        if(filename.find(uppercase_word) == string::npos) {
            continue;
        }

        out_files.push_back(p);
    }
}

template<typename T>
void get_filenames_containing_words(const vector <fs::path> &files, vector<fs::path> &out_files, T word) {
    filter_files_by_filename(files, out_files, word);
}

template<typename T, typename... Args>
void get_filenames_containing_words(const vector <fs::path> &files, vector<fs::path> &out_files, T word, Args... word_list) {
    filter_files_by_filename(files, out_files, word);

    get_filenames_containing_words(files, out_files, word_list...);

}

template<typename... Args>
DeviceTexture* get_texture(const vector<fs::path> &all_image_files, DeviceTextureLoader& texture_loader, Args... word_list) {
    vector<fs::path> filtered_image_files;
    get_filenames_containing_words(all_image_files, filtered_image_files, word_list...);

    if(filtered_image_files.empty()) {
        return nullptr;
    }

    return texture_loader.load(filtered_image_files[0]);
}

DeviceMaterial DeviceMaterialLoader::load(std::string_view directory) {
    if (!fs::exists(directory)) {
        cerr << "Directory '" << directory << "' does not exist." << endl;
        exit(1);
    }

    auto all_image_files = get_image_files(directory);

    auto diffuse_texture = get_texture(all_image_files, m_texture_loader, "color", "clr", "albedo", "diffuse");
    auto normal_texture = get_texture(all_image_files, m_texture_loader, "normal");
    auto roughness_texture = get_texture(all_image_files, m_texture_loader, "roughness");

    DeviceMaterial material{diffuse_texture};
    material.set_roughness_map(roughness_texture);
    material.set_normal_map(normal_texture);

    return material;
}
