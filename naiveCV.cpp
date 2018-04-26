#include "opencv2/core/core.hpp"
// #include "opencv2/imgproc/imgproc.hpp"
#include<iostream>
using namespace cv;
using namespace std;
int main(){
    cv::Mat_<uchar> canvas = cv::Mat::zeros(100, 100, CV_8UC1);
    cout << cv::sum(canvas)[0]<<endl;

    return 0;
}
