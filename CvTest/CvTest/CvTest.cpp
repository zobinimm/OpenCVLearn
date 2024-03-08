#pragma region Test1
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 读取图像
    cv::Mat image = cv::imread("your_image_path.jpg");

    if (image.empty()) {
        std::cerr << "Error: Unable to read the image." << std::endl;
        return -1;
    }

    // 读取要识别的图像
    cv::Mat templateImage = cv::imread("your_template_image.jpg");

    if (templateImage.empty()) {
        std::cerr << "Error: Unable to read the template image." << std::endl;
        return -1;
    }

    // 使用模板匹配进行图像识别
    cv::Mat result;
    cv::matchTemplate(image, templateImage, result, cv::TM_CCOEFF_NORMED);

    // 设置阈值，可以根据具体情况调整
    double threshold = 0.8;
    cv::threshold(result, result, threshold, 1.0, cv::THRESH_BINARY);

    // 寻找匹配位置
    std::vector<cv::Point> locations;
    cv::findNonZero(result, locations);

    // 在原图上标识匹配位置
    for (const cv::Point& loc : locations) {
        cv::Rect rect(loc.x, loc.y, templateImage.cols, templateImage.rows);
        cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2); // 用绿色矩形框标识
    }

    // 显示结果
    cv::imshow("Result", image);
    cv::waitKey(0);

    return 0;
}

#pragma endregion


#pragma region Test2
//#include <opencv2/opencv.hpp>
//
//int main() {
//    // 读取原始图像
//    cv::Mat img = cv::imread("your_image_path.jpg");
//
//    if (img.empty()) {
//        std::cerr << "Error: Unable to read the image." << std::endl;
//        return -1;
//    }
//    cv::Mat imgShow = cv::imread("your_image_path.jpg");
//
//    // 读取模板图像
//    cv::Mat templateImg = cv::imread("your_template_image.jpg");
//
//    if (templateImg.empty()) {
//        std::cerr << "Error: Unable to read the template image." << std::endl;
//        return -1;
//    }
//
//    // 转换颜色空间
//    cv::cvtColor(img, img, cv::COLOR_BGR2HSV);
//    cv::cvtColor(templateImg, templateImg, cv::COLOR_BGR2HSV);
//
//    // 通道分离
//    std::vector<cv::Mat> imgChannels, templateChannels;
//    cv::split(img, imgChannels);
//    cv::split(templateImg, templateChannels);
//
//    // 使用模板匹配
//    cv::Mat result;
//    cv::matchTemplate(imgChannels[2], templateChannels[2], result, cv::TM_CCOEFF_NORMED);
//
//    // 设置阈值，可以根据实际情况调整
//    double threshold = 0.8;
//    cv::threshold(result, result, threshold, 1.0, cv::THRESH_BINARY);
//
//    // 寻找匹配位置
//    std::vector<cv::Point> locations;
//    cv::findNonZero(result, locations);
//
//    // 在原图上标识匹配位置
//    for (const cv::Point& loc : locations) {
//        cv::Rect rect(loc.x, loc.y, templateImg.cols, templateImg.rows);
//        cv::rectangle(imgShow, rect, cv::Scalar(0, 255, 0), 2); // 用绿色矩形框标识
//    }
//
//    // 显示结果
//    cv::imshow("Result", imgShow);
//    cv::waitKey(0);
//
//    return 0;
//}
#pragma endregion

#pragma region Test3
//#include <opencv2/opencv.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <iostream>
//#include <Windows.h>  // Windows平台需要包含此头文件
//
//int main() {
//    // 创建屏幕捕获对象
//    cv::VideoCapture screenCapture(0);  // 0表示默认摄像头，也可以根据需要更改
//
//    if (!screenCapture.isOpened()) {
//        std::cerr << "Error: Unable to open screen capture." << std::endl;
//        return -1;
//    }
//
//    // 读取要识别的图像
//    cv::Mat templateImage = cv::imread("your_template_image.jpg");
//
//    if (templateImage.empty()) {
//        std::cerr << "Error: Unable to read the template image." << std::endl;
//        return -1;
//    }
//
//    // 初始化ORB检测器
//    cv::Ptr<cv::ORB> orb = cv::ORB::create();
//
//    // 检测特征点并计算描述符
//    std::vector<cv::KeyPoint> keypoints1, keypoints2;
//    cv::Mat descriptors1, descriptors2;
//
//    orb->detectAndCompute(templateImage, cv::noArray(), keypoints1, descriptors1);
//
//    while (true) {
//        // 从屏幕捕获图像
//        cv::Mat frame;
//        screenCapture >> frame;
//
//        if (frame.empty()) {
//            std::cerr << "Error: Unable to capture frame." << std::endl;
//            break;
//        }
//
//        // 检测特征点并计算描述符
//        orb->detectAndCompute(frame, cv::noArray(), keypoints2, descriptors2);
//
//        // 使用暴力匹配器进行特征匹配
//        cv::BFMatcher matcher(cv::NORM_HAMMING);
//        std::vector<cv::DMatch> matches;
//        matcher.match(descriptors1, descriptors2, matches);
//
//        // 筛选匹配点
//        double minDist = 100.0;
//        for (const cv::DMatch& match : matches) {
//            if (match.distance < minDist) {
//                minDist = match.distance;
//            }
//        }
//
//        std::vector<cv::DMatch> goodMatches;
//        for (const cv::DMatch& match : matches) {
//            if (match.distance < 2 * minDist) {
//                goodMatches.push_back(match);
//            }
//        }
//
//        // 在屏幕图像上标识匹配点
//        cv::Mat imgMatches;
//        cv::drawMatches(templateImage, keypoints1, frame, keypoints2, goodMatches, imgMatches);
//
//        // 显示结果
//        cv::imshow("Matches", imgMatches);
//
//        // 检测按键，按ESC键退出
//        if (cv::waitKey(30) == 27) {
//            break;
//        }
//    }
//
//    return 0;
//}
#pragma endregion