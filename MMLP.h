/*
    Multilayer Perceptron library for ESP32

    (c) 2021 Lesept
    contact: lesept777@gmail.com

    Permission is hereby granted, free of charge, to any person obtaining
    a copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
    OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
    ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
    OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef MLP_h
#define MLP_h

#include <Arduino.h>
#include "Matrix.h"
#include "FS.h"
#include <LITTLEFS.h>

#define MAX_LAYERS   10      // Maximum number of layers
#ifndef MAX_INPUT
#define MAX_INPUT    70      // Maximum number of neurons in input layer
#endif

// Heuristics options: set them if...
#define H_INIT_OPTIM      0x01  // if you initialize optimize
#define H_CHAN_WEIGH      0x02  // for brand new random weights
#define H_MUTA_WEIGH      0x04  // to slightly change the weights
#define H_CHAN_LRLIN      0x08  // to linearly change the learning rate
#define H_CHAN_LRLOG      0x10  // to change the learning rate log scale
#define H_CHAN_SGAIN      0x20  // to change the sigmoid gain
#define H_CHAN_MOMEN      0x40  // to change the momentum
#define H_SHUF_DATAS      0x80  // to shuffle the dataset
#define H_ZERO_WEIGH     0x100  // to force low weights to 0
#define H_STOP_TOTER     0x200  // stop optimization if test + train Error < threshold 
#define H_SELE_WEIGH     0x400  // select best weights over 10 random sets
#define H_FORC_S_G_D     0x800  // force stochastic gradient descent for faster optimization
#define H_REG1_WEIGH    0x1000  // use L1 weight regularization
#define H_REG2_WEIGH    0x2000  // use L2 weight regularization
#define H_BEST_ETA      0x4000  // search for best learning rate each epoch
#define H_LABL_SMOOT    0x8000  // for label smoothing
#define H_GRAD_CLIP    0x10000  // for gradient clipping
#define H_GRAD_SCALE   0x20000  // for gradient scaling

// 9 activation functions
enum ACTIVATION {
  RELU,     // 0
  SIGMOID,  // 1
  TANH,     // 2
  SOFTMAX,  // 3
  SIGMOID2, // 4
  ID,       // 5
  LEAKYRELU,// 6
  ELU,      // 7
  SELU,     // 8
  RELU6     // 9
};

// 3 cost functions
enum COST {
  MSE,            // Mean Square Error
  CROSSENTROPY,   // Cross Entropy
  LOGLIKELIHOOD   // Log Likelihood
};

typedef std::vector<MLMatrix<float> > Tensor;

class MLP
{
  public:
// Constructor
    MLP(const int*, const int, const int = 1);
    ~MLP();

// Hyper parameters
    // learning rate and momentum
    void setHyper (const float = 0.5f, const float = 0.1f);
    // parameters for optimization of training
    void displayHeuristics ();
    void setHeuristics (uint32_t);
    void setHeurShuffleDataset (bool);
    void setHeurZeroWeights (bool, float);
    void setHeurRegulL1 (bool, float = 1.0f);
    void setHeurRegulL2 (bool, float = 1.0f);
    void setHeurChangeMomentum (bool, float = 0.1f, float = 1.5f);
    void setHeurChangeGain (bool, float = 0.5f, float = 2.0f);
    void setHeurInitialize (bool);
    void setHeurGradScale (float);
    void setHeurGradClip (float);

    // Activation functions of each layer
    void setActivations (const int *);
    void setCost (const uint8_t);

// Training parameters
    void setVerbose   (uint8_t);
    void setEpochs    (int);
    void setBatchSize (int);
    void setMomentum  (float);
    void setEta       (float);
    void setEtaRange (float, float);
    void setGain      (float);

    int   getEpochs    ();
    int   getBatchSize ();
    float getMomentum  ();
    float getEta       ();
    float getGain      ();
    float getAnneal    ();
    int   getNeuronNumbers (int);

// Dataset functions
    void setTrainTest (float, float, float);
    void createDataset (const std::vector<std::vector<float> > , std::vector<std::vector<float> > &, const int);
    void shuffleDataset (std::vector<std::vector<float> > &, std::vector<std::vector<float> > &, uint16_t, uint16_t);
    void createDatasetFromArray (std::vector<std::vector<float> > &, std::vector<std::vector<float> > &, const float *, const float *, const int);
    void createDatasetFromVector (std::vector<std::vector<float> > &, std::vector<std::vector<float> > &, const std::vector<float>, const std::vector<float>);
    void normalizeDataset (std::vector<std::vector<float> > &, std::vector<std::vector<float> > &, const uint8_t = 0);
    int  readCsvFromSpiffs (const char* const, std::vector<std::vector<float> >&, std::vector<std::vector<float> >&);

// Train and prediction functions
    void run (std::vector<std::vector<float> >, std::vector<std::vector<float> >, int, int, float);
    MLMatrix<float> predict (MLMatrix<float>);
    uint32_t estimateDuration (int);

// Perceptron functions
    MLMatrix<float> forward (MLMatrix<float>, bool = true);
    float error (MLMatrix<float>, MLMatrix<float>) const;
    void  backward (std::vector<MLMatrix<float> > &, std::vector<MLMatrix<float> > &, const MLMatrix<float>, const MLMatrix<float>, const int);
    void  update (std::vector<MLMatrix<float> > &, std::vector<MLMatrix<float> >, std::vector<MLMatrix<float> >, std::vector<MLMatrix<float> > &, std::vector<MLMatrix<float> >, std::vector<MLMatrix<float> >, const int);
    float testNet(const std::vector<std::vector<float> >, const std::vector<std::vector<float> >, const uint16_t, const uint16_t, const bool);

// Weights functions
    void randomWeights (float = -0.5f, float = 0.5f);
    void normalize (MLMatrix<float> &, const uint8_t);
    void deNorm (MLMatrix<float> &, const uint8_t);
    void searchBestWeights (std::vector<std::vector<float> >, std::vector<std::vector<float> >);
    void saveWeights ();
    void restoreWeights ();
    float regulL1Weights();
    float regulL2Weights();
    int numberOfWeights();
    void displayWeights();
    float getWeight (int, int, int);
    int setWeight (int, int, int, float);
    void statWeights();
    float meanWeights();
    float stdevWeights (float);

// Misc functions
    int  size () const;
    void displayNetwork ();
    void netSave (const char* const);
    bool netLoad (const char* const);

  private:

// Initial values of parameters
    float _eta       = 0.5f;
    float _eta0      = _eta;
    float _momentum  = 0.1f;
    float _gain      = 1.0f;
    float _anneal    = 1.0f;
    float _trainTest = 0.8f;

    float _currError = 100.0f;
    float _etaMin    = 0.001f;
    float _minError  = 1000000.0f;
    float _wmin      = -0.5f;
    float _wmax      =  0.5f;
    float _zeroThreshold = 0.002f;
    float _logLRmax  = -1.0f;
    float _logLRmin  = -4.0f;
    float _minAlpha  = 0.1f;
    float _maxAlpha  = 1.5f;
    float _minGain   = 0.5f;
    float _maxGain   = 2.0f;

    float _lambdaRegulL1  = 0.5f;
    float _lambdaRegulL2  = 0.5f;
    float _gradScale = 1.0f;
    float _gradClipValue  = 0.75f;

    float rTrain = 4.0f / 6.0f;
    float rValid = 1.0f / 6.0f;
    float rTest  = 1.0f / 6.0f;
    uint8_t _norm    = 0;
    uint8_t _nDots   = 0;

// Private variables
    uint8_t _nLayers, _nInputs, _nClasses;
    uint8_t _neurons[MAX_LAYERS];
    uint8_t _activations[MAX_LAYERS];
    uint8_t _verbose;
    uint8_t _cost = 0;
    uint16_t _maxEpochs;
    uint16_t _nTrain, _nValid, _nTest;
    float _stopError, _prevMinError, _validError;
    float _prevError, _firstError;
    int _nData, _batchSize, _lastBestEpoch = 0;
    bool _bestEta  = false;
    bool _firstRun = true;
    bool _datasetSplit = false;
    bool _enSoftmax = false;

// Storage vectors
    std::vector<float> _xmin;
    std::vector<float> _xmax;
    std::vector<float> _ymin;
    std::vector<float> _ymax;
    std::vector<float> _xMean;
    std::vector<float> _xStdev;
    std::vector<float> _yMean;
    std::vector<float> _yStdev;

// Network parameters (weights)
    std::vector<MLMatrix<float> > Weights;
    std::vector<MLMatrix<float> > Biases;
    std::vector<MLMatrix<float> > _a;
    // std::vector<MLMatrix<float> > _z;

// Saved values (save / restore network)
    std::vector<MLMatrix<float> > Weights_save;
    std::vector<MLMatrix<float> > Biases_save;
    float _eta_save, _momentum_save, _gain_save;

// Private functions: activations
    float ReLu (const float);
    float dReLu (const float);
    float ReLu6 (const float);
    float dReLu6 (const float);
    float Sigmoid (const float);
    float dSigmoid (const float);
    float Sigmoid2 (const float);
    float dSigmoid2 (const float);
    float Tanh (const float);
    float dTanh (const float);
    float Id (const float);
    float dId (const float);
    float SeLu (const float);
    float dSeLu (const float);
    float LeakyReLu (const float);
    float dLeakyReLu (const float);
    float ELu (const float);
    float dELu (const float);
    MLMatrix<float> SoftMax (const MLMatrix<float>);

// Private functions
    float CrossEntropy (const MLMatrix<float>, const MLMatrix<float>);
    MLMatrix<float> Mse (MLMatrix<float>, MLMatrix<float>);
    MLMatrix<float> LogLikelihood(MLMatrix<float>);
    MLMatrix<float> activation (MLMatrix<float>, const uint8_t);
    MLMatrix<float> dActivation (MLMatrix<float>, const uint8_t);

    void heuristics (int, int, bool);
    void searchEta (MLMatrix<float>, MLMatrix<float>, float, std::vector<MLMatrix<float> >, std::vector<MLMatrix<float> >, std::vector<MLMatrix<float> >, std::vector<MLMatrix<float> >);
    int   readIntFile (File);
    float readFloatFile (File);
    void initWeights ();
    MLMatrix<float> predict_nonorm (MLMatrix<float>);

    // typedef float (MLP::*Act) (const float);
    // Act Activation[3] = {&MLP::ReLu, &MLP::Sigmoid, &MLP::Tanh};
    // Act dActivation[3] = {&MLP::dReLu, &MLP::dSigmoid, &MLP::dTanh};
    char ActivNames[9][10] = {"RELU", "SIGMOID", "TANH", "SOFTMAX",
                              "SIGMOID2", "IDENTITY", 
                              "LEAKYRELU", "ELU", "SELU"};

// Booleans for the heuristics
    uint32_t _heuristics     = 0;
    bool     _initialize     = true;
    bool     _changeWeights  = false;
    bool     _mutateWeights  = false;
    bool     _changeLRlin    = false;
    bool     _changeLRlog    = false;
    bool     _changeMom      = false;
    bool     _changeGain     = false;
    bool     _shuffleDataset = false;
    bool     _zeroWeights    = false;
    bool     _stopTotalError = false;
    bool     _selectWeights  = false;
    bool     _forceSGD       = false;
    bool     _regulL1        = false;
    bool     _regulL2        = false;
    bool     _labelSmoothing = false;
    bool     _gradClip       = false;
    bool     _gradScaling    = false;
};

inline float halfSquare (const float x) { return 0.5f * pow(x, 2); }
inline float minusLog (const float x) { return -log(x + 1.0e-15); }
inline float rand01 () { return float(random(10000)) / 10000.0f; }

#endif
