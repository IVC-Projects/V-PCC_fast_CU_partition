#include <fdeep/fdeep.hpp>
#include <iostream>
#include <Buffer.h>
#include <EncoderLib/EncModeCtrl.h>
#include <fstream>
#include <LightGBM/prediction_early_stop.h>
#include <LightGBM/boosting.h>

using namespace std;
typedef std::pair<int, int> partsize;
static const int            MAX_CANNY_SIZE = 518400;   ///< log2(CTUSize)

class PartitionPrediction
{
public:
  PartitionPrediction();
  ~PartitionPrediction();
  //tissier
  PartitionPrediction(std::string model_str, int qp, bool intra);

  void cannyDetection(Pel *buf, int width, int height, int stride, double *list, int qp);
  double calcLaplace(Pel *yPxl, int width, int height, int stride);

  double calcLaplace2(Pel *yPxl, int x, int y, int width, int height, int stride);
  vector<partsize> getPartsizeList();
  EncTestModeType stringToModeType(string s);
  void            initializeModels(std::string modelFolder);
  void            predict_once(double *input, double *output, partsize size);
  void            splitByML(double *list);
  void            setStride(int x);
  int            getStride();
  uint8_t *        splitByML(int width, int height, double *input);
  int             getInverseSplit(EncTestModeType mode);
  double edgeList[MAX_CANNY_SIZE];
  std::unique_ptr<fdeep::model> model;

private:
  /*PyObject * pMod       = NULL;
  PyObject * load_model = NULL;
  PyObject * predict    = NULL;
  PyObject * pParm      = NULL;
  PyObject * pArgs      = NULL;*/
  std::vector<partsize>                    partsize_list;
  std::map<partsize, LightGBM::Boosting *> models;
  std::map<partsize, LightGBM::Boosting *> modelsP;
  int                                      cannyStride;

  int qp;
};
