
#include <iostream>
#include <fstream>
#include "PartitionPredict.h"
#include <string>
//using namespace cv;
using namespace std;

PartitionPredict::PartitionPredict(string filename) {
  //std::string   fileName = "D:/wsr/dct_model/model_";
  //if (op == 1)
  //  fileName += "split_qp" + std::to_string(qp) + "_v2.txt";
  //else
  //  fileName += "direction_qp" + std::to_string(qp) + "_v2.txt";
  //std::cout << fileName << "\n";
  //LGBM_BoosterCreateFromModelfile(fileName.c_str(), &p, &this->handle);
  std::ifstream model_file;

  const char* charFilename = filename.c_str();
  model_file.open(filename, std::ifstream::in);
  
  std::string         model_content((std::istreambuf_iterator<char>(model_file)), std::istreambuf_iterator<char>());
  unsigned long       size_t = model_content.length();
  const char*         cstr   = model_content.c_str();
  this->model = LightGBM::Boosting::CreateBoosting("gbdt", charFilename);
  this->model->LoadModelFromString(cstr, size_t);
  model_file.close();
}

PartitionPredict::PartitionPredict(string filename, int op)
{
  //rtmodel = ml::RTrees::load(filename);
}

PartitionPredict::~PartitionPredict()
{
   
}

