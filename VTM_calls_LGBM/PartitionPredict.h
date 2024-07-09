#include <iostream>
#include <EncoderLib/EncModeCtrl.h>
#include <LightGBM/c_api.h>
#include <LightGBM/boosting.h>
//#include <opencv2/opencv.hpp>
//using namespace cv;
class PartitionPredict
{
public:
  PartitionPredict(std::string filename);
  PartitionPredict(std::string filename, int op);
  ~PartitionPredict();

  

//private:
  LightGBM::Boosting* model;
  //cv::Ptr<cv::ml::RTrees>     rtmodel;
};
