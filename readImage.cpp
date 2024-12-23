#include <opencv2/opencv.hpp>
#include <iostream>
// g++ -o display_image readImage.cpp `pkg-config --cflags --libs opencv4`
int main() {
    // Read the image file
    cv::Mat image = cv::imread("unet.png", cv::IMREAD_COLOR);
    

    // Check for failure
    if (image.empty()) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    // Display the image
    cv::imshow("Display window", image);

    // Wait for a keystroke in the window
    cv::waitKey(0);
    return 0;
}
