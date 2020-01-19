#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "CameraCtl.hpp"
#include <opencv2/imgproc.hpp>
using namespace cv;
using namespace dnn;
constexpr const char *image_path = "2.jpg";//待检测图片
constexpr const char *darknet_cfg = "../darknet.cfg";//网络文件
constexpr const char *darknet_weights = "../darknet.weights";//训练模型
const std::vector<std::string> class_labels = {"car"};//类标签
float confidenceThreshold=0.01;

void car_detection(Mat img,Net net)
{
    // 加载标签集
    std::vector<std::string> classLabels = class_labels;
    // 读取待检测图片
    
    cv::Mat blob = cv::dnn::blobFromImage(img,1.0/255.0,{416,416},0.1,true);
    net.setInput(blob,"data");
    // 检测
    cv::Mat detectionMat = net.forward();// 6 845 1 W x H x C
    // 获取网络每层的用时并获取总用时
    std::vector<double> layersTimings;
    double freq = cv::getTickFrequency() / 1000;
    double time = net.getPerfProfile(layersTimings) / freq;
    std::ostringstream ss;
    ss << "detection time: " << time << " ms";
    // 绘制总用时至原始图片
    cv::putText(img, ss.str(), cv::Point(20, 20), 0, 0.5, cv::Scalar(0, 0, 255));
    // 遍历所有结果集
    for(int i = 0;i < detectionMat.rows;++i){
        const int probability_index = 5;
        const int probability_size = detectionMat.cols - probability_index;
        float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);
        size_t objectClass = std::max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
        float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);

        // 比较置信度并绘制满足条件的置信度
        if (confidence > confidenceThreshold)
        {
            float x = detectionMat.at<float>(i, 0);
            float y = detectionMat.at<float>(i, 1);
            float width = detectionMat.at<float>(i, 2);
            float height = detectionMat.at<float>(i, 3);

            int xLeftBottom = static_cast<int>((x - width / 2) * img.cols);
            int yLeftBottom = static_cast<int>((y - height / 2) * img.rows);
            int xRightTop = static_cast<int>((x + width / 2) * img.cols);
            int yRightTop = static_cast<int>((y + height / 2) * img.rows);

            cv::Rect object(xLeftBottom, yLeftBottom,xRightTop - xLeftBottom,yRightTop - yLeftBottom);//x y w h
            cv::rectangle(img, object, cv::Scalar(0, 0, 255), 2, 8);

            // 判断类id是否符合标签范围并绘制该标签，也就是矩阵的下标索引
            if (objectClass < classLabels.size())
            {
                cv::String label = cv::String(classLabels[objectClass]) + ": " + std::to_string(confidence);
                int baseLine = 0;
                cv::Size labelSize = cv::getTextSize(label,cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                cv::rectangle(img, cv::Rect(cv::Point(xLeftBottom, yLeftBottom),cv::Size(labelSize.width, labelSize.height + baseLine)),cv::Scalar(255, 255, 255), cv::FILLED);
                cv::putText(img, label, cv::Point(xLeftBottom, yLeftBottom + labelSize.height),cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            }
        }
    }
    // 显示图片
    cv::imshow("Darknet",img);
    cv::waitKey(1);
}


int main(int argc, char **argv)
{
    //if(argc<2){
    //    std::cout<<"Please input picture path."<<std::endl;
    //}
    //std::string pic_path = argv[1];
    //cv::Mat img = cv::imread(pic_path);
    // 加载模型
    Net net = readNetFromDarknet(darknet_cfg,darknet_weights);
   //car_detection(img,net);
    std::string outputVideoPah = "test.avi";
    Size sWH = Size(1440, 108);
    //ideoWriter outputVideo;
    //outputVideo.open(outputVideoPath, CAP_OPENCV_MJPEG, 0.0, sWH);
    CameraCtl camCtl;
    camCtl.startGrabbing();
    camCtl.setGainMode(CONTINUOUS);
    camCtl.getGainMode(true);
    //camCtl.setGain(5);
    camCtl.getFrameRate(true);
    camCtl.getGain(true);
    camCtl.setAutoExposure(CONTINUOUS);
    camCtl.setExposureTime(10000);
    camCtl.getAutoExposure(true);
    camCtl.setFrameRate(15);
    camCtl.getFrameRate();
    printf("Press any key to exit.\n");
    namedWindow("Image");
    char key=0;
    while (true) {
        Mat img = camCtl.getOpencvMat();
        imshow("Image", img);
        car_detection(img,net);
        //outputVideo<<img;
        key=waitKey(1);
        if (key == 27)
            break;
        else if(key==' ')
            waitKey(0);
    }
    //outputVideo.release();
    destroyAllWindows();
    return 0;
}

