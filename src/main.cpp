#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "HsvHistogramFeatureVectorComputer.hpp"
#include "JsonFeatureStorage.hpp"
#include "ManhattanFeatureDistanceComputer.hpp"


const std::string CMD_ADD_IMAGE = "add_image";
const std::string CMD_ADD_DIRECTORY = "add_directory";
const std::string CMD_FIND_SIMILAR = "find_similar";

int main(int argc, char** argv) {
    if (argc == 1) {
        std::cerr << "Please specify a command.\n";
        return -1;
    }

    std::string command = argv[1];
    std::string storage_fname = "storage1.json";
    bool save_at_exit = false;
    cbir::HsvHistogramFeatureVectorComputer feature_computer{};
    cbir::ManhattanFeatureDistanceComputer<float, cbir::HSV_Features_D> distance_computer{};
    cbir::JsonFeatureStorage<float, cbir::HSV_Features_D> storage{storage_fname, feature_computer, distance_computer};

    if (command == CMD_ADD_IMAGE) {
        if (argc != 3) {
            std::cerr << "Command usage: " << argv[0] << " " << CMD_ADD_IMAGE << " <image_path>\n";
            return -1;
        }
        
        std::filesystem::path image_path(argv[2]);
        if (!std::filesystem::exists(image_path)) {
            std::cerr << "Specified image doesn't exist.\n";
            return -1;
        }

        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Failed to read specified image.\n";
            return -1;
        }

        storage.add_image(image_path.filename(), image);
        save_at_exit = true;

    } else if (command == CMD_ADD_DIRECTORY) {
        if (argc != 3) {
            std::cerr << "Command usage: " << argv[0] << " " << CMD_ADD_DIRECTORY << " <directory_path>\n";
            return -1;
        }
        
        std::filesystem::path dir_path(argv[2]);
        if (!std::filesystem::exists(dir_path) || !std::filesystem::is_directory(dir_path)) {
            std::cerr << "Specified directory doesn't exist.\n";
            return -1;
        }

        for (const auto& image_entry: std::filesystem::directory_iterator(dir_path)) {
            const auto image_path = image_entry.path();
            
            cv::Mat image = cv::imread(image_path);
            if (image.empty()) {
                std::cerr << "Failed to read specified image.\n";
                continue;
            }

            storage.add_image(image_path.filename(), image);
            save_at_exit = true;
        }
    } else if (command == CMD_FIND_SIMILAR) {
        if (argc != 4) {
            std::cerr << "Command usage: " << argv[0] << " " << CMD_FIND_SIMILAR << " <file_path> <num_finds>\n";
            return -1;
        }
        
        const std::filesystem::path file_name(argv[2]);
        const std::string key = file_name.filename();
        const size_t num_finds = std::stoll(argv[3]);

        const auto result = storage.find_nearest(key, num_finds);
        std::cout << "Found similar keys"
            << (result.empty() ? "." : ":") << "\n";
        
        size_t result_counter = 1;
        for (const auto& [key, distance]: result) {
            std::cout << result_counter << ". " << key << "\n";
            ++result_counter;
        }
    }

    std::cout << "Done processing the command.\n";
    if (save_at_exit) {
        storage.save();
    }
}
