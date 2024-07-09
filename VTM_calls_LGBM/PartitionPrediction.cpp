#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream> 
#include "PartitionPrediction.h"
#include <string>
#include <algorithm>

using namespace cv;
using namespace std;

bool com(double a, double b)
{
  return a > b;
}
const double EPS = 0.0000001;

PartitionPrediction::PartitionPrediction()
{ 
    //intra
  this->partsize_list = { /*std::make_pair(4, 8),   std::make_pair(8, 4), std::make_pair(64, 64),*/
    std::make_pair(8, 32), std::make_pair(16, 8),
    std::make_pair(8, 8),  /* std::make_pair(4, 32),*/  std::make_pair(32, 32), 
    std::make_pair(16, 16), std::make_pair(32, 8), /* std::make_pair(16, 4),  std::make_pair(32, 4),
    std::make_pair(4, 16), */ std::make_pair(32, 16), std::make_pair(8, 16),  std::make_pair(16, 32)
  };
    
}

PartitionPrediction::PartitionPrediction(std::string model_str, int qp, bool intra)
{
  if (intra)
  {
    this->partsize_list = { std::make_pair(4, 8),   std::make_pair(64, 64), std::make_pair(8, 32),
                            std::make_pair(16, 8),  std::make_pair(8, 8),   std::make_pair(4, 32),
                            std::make_pair(32, 32), std::make_pair(8, 4),   std::make_pair(16, 16),
                            std::make_pair(32, 8),  std::make_pair(16, 4),  std::make_pair(32, 4),
                            std::make_pair(4, 16),  std::make_pair(32, 16), std::make_pair(8, 16),
                            std::make_pair(16, 32) };
  }
  else
  {
    this->partsize_list = {
      std::make_pair(128, 128), std::make_pair(128, 64), std::make_pair(64, 128), std::make_pair(128, 32),
      std::make_pair(32, 128),  std::make_pair(128, 16), std::make_pair(16, 128), std::make_pair(16, 64),
      std::make_pair(32, 64),   std::make_pair(4, 64),   std::make_pair(64, 16),  std::make_pair(64, 32),
      std::make_pair(64, 4),    std::make_pair(8, 64),   std::make_pair(4, 8),    std::make_pair(64, 64),
      std::make_pair(64, 8),    std::make_pair(8, 32),   std::make_pair(16, 8),   std::make_pair(8, 8),
      std::make_pair(4, 32),    std::make_pair(32, 32),  std::make_pair(8, 4),    std::make_pair(16, 16),
      std::make_pair(32, 8),    std::make_pair(16, 4),   std::make_pair(32, 4),   std::make_pair(4, 16),
      std::make_pair(32, 16),   std::make_pair(8, 16),   std::make_pair(16, 32)
    };
  }
  this->qp = qp;
  cout << model_str << endl;
  this->model = std::make_unique<fdeep::model>(fdeep::load_model(model_str));
}


PartitionPrediction::~PartitionPrediction()
{
  // free(this->model);
}

vector<partsize> PartitionPrediction::getPartsizeList()
{
  return partsize_list;
}

void PartitionPrediction::cannyDetection(Pel *buf, int width, int height, int stride, double *edgeList, int qp)
{
  Mat edges;
  Mat img(height, width, CV_8UC1, Scalar(0));   //无符号8位
  //vector<int> imgY;
  for (int i = 0; i < height; i++)
  {
    for (int j = 0; j < width; j++)
    {
      int val             = *(buf + i * stride + j);
      img.at<uchar>(i, j) = static_cast<uchar>(val);
      //imgY.push_back(val);
    }
  }
  /*sort(imgY.begin(), imgY.end());
  int midSize = imgY.size() / 2;
  nth_element(imgY.begin(), imgY.begin() + midSize, imgY.end());
  int midNum = imgY[midSize];
  float sigma  = 0.33;
  int   lowTH  = (1- sigma) * midNum;
  int highTH = (1 + sigma) * midNum > 255 ? 255 : (1 + sigma) * midNum;*/

  int lowTH = 50;
  int highTH = 100;
 
  Canny(img, edges, lowTH, highTH);
  for (int i = 0; i < edges.rows; i++)
  {
    for (int j = 0; j < edges.cols; j++)
    {
      if (edges.at<uchar>(i, j) == 255)
      {
        // 获取边缘强度值
        // int edgeStrength = (int) img.at<uchar>(i, j);
        int idx = i / 4 * (width / 4) + j / 4;
        edgeList[idx] += 1;
      }
    }
  }
}

void PartitionPrediction::setStride(int x) {
  this->cannyStride = x;
}

int PartitionPrediction::getStride()
{
  return this->cannyStride;
}

void PartitionPrediction::initializeModels(std::string modelFolder)
{
  for (int i = 0; i < partsize_list.size(); ++i)
  {
    // load model file
    std::ifstream model_file;
    
    //dgc
    /*std::string filename = modelFolder + "//" + std::to_string(partsize_list[i].first) + "x"
                           + std::to_string(partsize_list[i].second) + ".txt";*/

    //tissier
    std::string filename = modelFolder + "ML_model/intra" + "/lgbm_" + std::to_string(partsize_list[i].first) + "x"
                           + std::to_string(partsize_list[i].second) + ".txt";

    const char *charFilename = filename.c_str();
    model_file.open(filename, std::ifstream::in);
    std::string   model_content((std::istreambuf_iterator<char>(model_file)), std::istreambuf_iterator<char>());
    unsigned long size_t     = model_content.length();
    const char *  cstr       = model_content.c_str();
    try
    {
      cout << charFilename << endl;
      models[partsize_list[i]] = LightGBM::Boosting::CreateBoosting("gbdt", charFilename);
    }
    catch (const std::exception &ex)
    {
      std::cerr << "Met Exceptions:" << std::endl;
      std::cerr << ex.what() << std::endl;
    }
    catch (const std::string &ex)
    {
      std::cerr << "Met Exceptions:" << std::endl;
      std::cerr << ex << std::endl;
    }
    catch (...)
    {
      std::cerr << "Unknown Exceptions" << std::endl;
    }
    try
    {
      cout << "hello" << endl;
      models[partsize_list[i]]->LoadModelFromString(cstr, size_t);
      cout << "byebye" << endl;
    }
    catch (const std::exception &ex)
    {
      std::cerr << ex.what() << std::endl;

    }
    model_file.close();
  }
}

void PartitionPrediction::predict_once(double *input, double *output, partsize size)
{
  LightGBM::PredictionEarlyStopConfig test;
  std::string                         classif = "multiclass";
  if ((size.first == 64 && size.second == 64) || (size.first == 8 && size.second == 4)
      || (size.first == 4 && size.second == 8))
  {
    classif = "binary";
  }
  const LightGBM::PredictionEarlyStopInstance early_stop = LightGBM::CreatePredictionEarlyStopInstance(classif, test);
  this->models[size]->Predict(input, output, &early_stop);
}

vector<int> findTopK(double arr[], int n) {
  vector<int> res;
  priority_queue<pair<double, int>> q;
  for (int i = 0; i < n; i++) {
    q.push(make_pair(arr[i], i));
  }
  while (!q.empty()) {
    res.push_back(q.top().second);
    q.pop();
  }
  return res;
}

double PartitionPrediction::calcLaplace(Pel *yPxl, int width, int height, int stride)
{
  yPxl += stride;
  double L90 = 0, L180 = 0;
  for (int i = 1; i < height - 1; i++)
  {
    for (int j = 1; j < width - 1; j++)
    {
      L90 += abs(*(yPxl - stride + j ) + *(yPxl + stride + j) - 2 * (*(yPxl + j)));
      L180 += abs(*(yPxl + j - 1) + *(yPxl + j + 1) - 2 * (*(yPxl + j)));
    }
    yPxl += stride;
  }
  if (L90 == 0)
    L90 = 0.0001;
  return L180 / L90;
}

double PartitionPrediction::calcLaplace2(Pel *yPxl,int x, int y, int width, int height, int stride)
{
  yPxl += stride * y;
  double L90 = 0, L180 = 0;
  for (int i = 0; i < height/* - 1*/; i++)
  {
    for (int j = 0; j < width/* - 1*/; j++)
    {
      if (y > 0 && y < height - 1)
        L90 += abs(*(yPxl + x + j - stride) + *(yPxl + x + j + stride) - 2 * (*(yPxl + x + j)));
      if (x > 0 && x < width - 1)
        L180 += abs(*(yPxl + x + j - 1) + *(yPxl + x + j + 1) - 2 * (*(yPxl + x + j)));
    }
    yPxl += stride;
  }
  if (L90 == 0)
    L90 = 0.0001;
  return L180 / L90;
}

uint8_t *PartitionPrediction::splitByML(int width, int height, double *input)
{
  std::map<partsize, vector<int>> resClass = { 
    { std::make_pair(4, 8), { 4, 5 } },
    { std::make_pair(64, 64), { 1, 5 } }, // intra
    //{ std::make_pair(64, 64), { 0, 1, 2, 3, 4, 5 } }, // inter
    { std::make_pair(8, 32), { 0, 3, 4, 5 } },
    { std::make_pair(16, 8), { 0, 2, 4, 5 } },
    { std::make_pair(8, 8), { 0, 4, 5 } },
    { std::make_pair(4, 32), { 3, 4, 5 } },
    { std::make_pair(32, 32), { 0, 1, 2, 3, 4, 5 } },
    { std::make_pair(8, 4), { 0, 5 } },
    { std::make_pair(16, 16), { 0, 1, 2, 3, 4, 5 } },
    { std::make_pair(32, 8), { 0, 2, 4, 5 } },
    { std::make_pair(16, 4), { 0, 2, 5 } },
    { std::make_pair(32, 4), { 0, 2, 5 } },
    { std::make_pair(4, 16), { 3, 4, 5 } },
    { std::make_pair(32, 16), { 0, 2, 3, 4, 5 } },
    { std::make_pair(8, 16), { 0, 3, 4, 5 } },
    { std::make_pair(16, 32), { 0, 2, 3, 4, 5 } },
    { std::make_pair(128, 128), { 0, 1, 4, 5 } },
    { std::make_pair(64, 16), { 0, 2, 3, 4, 5 } },
    { std::make_pair(16, 64), { 0, 2, 3, 4, 5 } },
    { std::make_pair(64, 32), { 0, 2, 3, 4, 5 } },
    { std::make_pair(32, 64), { 0, 2, 3, 4, 5 } },
  };

  //['CU_HORZ_SPLIT' 'CU_QUAD_SPLIT' 'CU_TRIH_SPLIT' 'CU_TRIV_SPLIT' 'CU_VERT_SPLIT' 'NO_SPLIT']
  double *              output = (double *) malloc(sizeof(double) * 6);
 
  uint8_t *splitDecision = new uint8_t[6];  
  for (int i = 0; i < 6; i++)
  {
    splitDecision[i] = 0; 
  }
  for (int i = 0; i < 6; i++)
  {
    output[i] = 0;
  }
  try
  {
    this->predict_once(input, output, make_pair(height, width));
  }
  catch (exception &e)
  {
    cout << e.what() << endl;
  }
  // binary的单独判断 intra
  //if (width == 64 && height == 64)
  //{
  //  //cout << " ----64x64 ---" << output[0] << endl;
  //  if (output[0] < 0.78)
  //  {
  //    splitDecision[1] = 1;
  //  }
  //}
  //else
  {
    vector<int> tmp  = resClass[make_pair(height, width)];
    vector<int> top3            = findTopK(output, tmp.size());
    splitDecision[tmp[top3[0]]] = 1;
    splitDecision[tmp[top3[1]]] = 1;
    // slow
    //splitDecision[tmp[top3[2]]] = 1;

   
    //fast
    if ((width == 32 && height == 32) || (width == 16 && height == 16))
    {
      splitDecision[tmp[top3[2]]] = 1;
    }

    /*if ((width == 32 && height == 32) || (width == 32 && height == 16) || (width == 16 && height == 32))
    {
      splitDecision[tmp[top3[2]]] = 1;
    }*/
   
  }
  
  delete[] output;
  return splitDecision;
}
 


EncTestModeType PartitionPrediction::stringToModeType(string s)
{
  if (s == "CU_HORZ_SPLIT")
  {
    return ETM_SPLIT_BT_H;
  }
  else if (s == "CU_VERT_SPLIT")
  {
    return ETM_SPLIT_BT_V;
  }
  else if (s == "CU_TRIH_SPLIT")
  {
    return ETM_SPLIT_TT_H;
  }
  else if (s == "CU_TRIV_SPLIT")
  {
    return ETM_SPLIT_TT_V;
  }
  else if (s == "CU_QUAD_SPLIT")
  {
    return ETM_SPLIT_QT;
  }
  return ETM_POST_DONT_SPLIT;
}


int PartitionPrediction::getInverseSplit(EncTestModeType mode)
{
  //['CU_HORZ_SPLIT' 'CU_QUAD_SPLIT' 'CU_TRIH_SPLIT' 'CU_TRIV_SPLIT' 'CU_VERT_SPLIT' 'NO_SPLIT']
  switch (mode)
  {
  case ETM_SPLIT_BT_H: return 0;
  case ETM_SPLIT_QT: return 1;
  case ETM_SPLIT_TT_H: return 2;
  case ETM_SPLIT_TT_V: return 3;
  case ETM_SPLIT_BT_V: return 4;
  default: return -1;
  }
}

