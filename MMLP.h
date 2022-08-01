/*
    Multilayer Perceptron library for ESP32

    (c) 2022 Lesept
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
#include "MatrixUT.hpp"
// #include "Matrix.h"
#include "FS.h"
#include <LITTLEFS.h>

#define MAX_LAYERS   10      // Maximum number of layers
#ifndef MAX_INPUT
#define MAX_INPUT    70      // Maximum number of neurons in input layer
#endif

// Heuristics options: set them if...
#define H_INIT_OPTIM        0x01  // if you initialize optimize
#define H_CHAN_WEIGH        0x02  // enable brand new random weights when needed
#define H_MUTA_WEIGH        0x04  // to slightly change the weights
#define H_CHAN_LRLIN        0x08  // to linearly change the learning rate
#define H_CHAN_LRLOG        0x10  // to change the learning rate log scale
#define H_CHAN_SGAIN        0x20  // to randomly change the sigmoid gain
#define H_CHAN_MOMEN        0x40  // to randomly change the momentum
#define H_SHUF_DATAS        0x80  // to shuffle the dataset
#define H_ZERO_WEIGH       0x100  // to force low weights to 0
#define H_STOP_TOTER       0x200  // stop optimization if test + train Error < threshold 
#define H_SELE_WEIGH       0x400  // select best weights over 30 random sets
#define H_INIT_XAVIE       0x800  // init weights with Xavier method
#define H_REG1_WEIGH      0x1000  // use L1 weight regularization
#define H_REG2_WEIGH      0x2000  // use L2 weight regularization
#define H_BEST_ETA        0x4000  // search for best learning rate each epoch
#define H_LABL_SMOOT      0x8000  // for label smoothing
#define H_GRAD_CLIP      0x10000  // clip gradient over threshold
#define H_GRAD_SCALE     0x20000  // for gradient scaling
#define H_CHAN_MOLIN     0x40000  // change the momentum throughout the epochs
#define H_DATA_SUBSE     0x80000  // begin training on a subset of the dataset
#define H_TOPK_PRUNE    0x100000  // prune network using Top-K method
#define H_TEST_PRUNE    0x200000  // prune inactive or low activity neurons at test phase
#define H_TRAI_PRUNE    0x400000  // prune inactive neurons during training
#define H_DROP_OUT      0x800000  // dropout: randomly remove neurons during forward prop (not good)
#define H_CHAN_BATSZ   0x1000000  // change the batch size during training
#define H_QUAD_LAYER   0x2000000  // use quadratic layers
#define H_DEEP_SHIFT   0x4000000  // use the Deep Shift algorithm
#define H_SKIP_CNECT   0x8000000  // use skip connections (not coded yet...)

// 9 activation functions
enum ACTIVATION {
  RELU,     // 0
  SIGMOID,  // 1
  TANH,     // 2
  SOFTMAX,  // 3
  SIGMOID2, // 4 (sigmoid with constant parts when close to +-1, quicker)
  ID,       // 5 (identity)
  LEAKYRELU,// 6 (returns 0.1 * x if x<0)
  ELU,      // 7
  SELU,     // 8
  RELU6,    // 9 (ReLu clipping at 6)
  SWISH     // 10
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
    // ~MLP();
    virtual ~MLP();

// Hyper parameters
// learning rate and momentum
    void setHyper (const float = 0.5f, const float = 0.1f);
// parameters for optimization of training
    void displayHeuristics     ();
    void setHeuristics         (uint32_t);
    void setHeurShuffleDataset (const bool);
    void setHeurZeroWeights    (const bool, const float = 0.15f);
    void setHeurRegulL1        (const bool, const float = 1.0f);
    void setHeurRegulL2        (const bool, const float = 1.0f);
    void setHeurChangeMomentum (const bool, const float = 0.1f, const float = 1.5f);
    void setHeurChangeGain     (const bool, const float = 0.5f, const float = 2.0f);
    void setHeurInitialize     (const bool);
    void setHeurGradScale      (const bool, const float = 1.0f);
    void setHeurGradClip       (const bool, const float);
    void setHeurPruning        (const bool, const float = 0.85f);
    void setHeurTopK           (const bool, const float);

// Activation functions of each layer
    void setActivations (const int *);
    void setCost        (const uint8_t);

// Training parameters
    void setVerbose    (const uint8_t);
    void setEpochs     (const int);
    void setBatchSize  (const int);
    void setMomentum   (const float);
    void setEta        (const float);
    void setEtaRange   (const float, const float);
    void setMomRange   (const float, const float);
    void setBSizeRange (float, float);
    void setGain       (const float);

    int   getEpochs    ();
    int   getBatchSize ();
    float getMomentum  ();
    float getEta       ();
    float getGain      ();
    int   getNeuronNumbers (int);

// Dataset functions
    void setTrainTest (float, float, float);
    void createDataset (MLMatrix<float> , MLMatrix<float> &, const int);
    void shuffleDataset (MLMatrix<float> &, MLMatrix<float> &, uint16_t, uint16_t);
    void createDatasetFromArray (MLMatrix<float> &, MLMatrix<float> &, const float *, const float *, const int);
    void createDatasetFromVector (MLMatrix<float> &, MLMatrix<float> &, const std::vector<float>, const std::vector<float>);
    void normalizeDataset (MLMatrix<float> &, MLMatrix<float> &, const uint8_t = 0);
    int  readCsvFromSpiffs (const char* const, MLMatrix<float> &, MLMatrix<float> &);

// Train and prediction functions
    void run (MLMatrix<float>, MLMatrix<float>, int, int, float);
    MLMatrix<float> predict (MLMatrix<float>);
    // MLMatrix<float> predict_nonorm (MLMatrix<float>);
    uint32_t estimateDuration (int);

// Perceptron functions
    MLMatrix<float> forward (MLMatrix<float>, bool = true);
    float error (MLMatrix<float>, MLMatrix<float>) const;
    void  backward (const MLMatrix<float>, const MLMatrix<float>, const int);
    void  update   (const int);
    float testNet  (const MLMatrix<float>, const MLMatrix<float>, const uint16_t, const uint16_t, const bool);

// Weights functions
    void  randomWeights     (float = -0.5f, float = 0.5f);
    void  normalize         (MLMatrix<float> &, const uint8_t);
    void  deNorm            (MLMatrix<float> &, const uint8_t);
    // void  deNorm            (std::vector<std::vector<float> > &, const uint8_t);
    void  searchBestWeights (const MLMatrix<float>, const MLMatrix<float>);
    void  saveWeights     ();
    void  restoreWeights  ();
    float regulL1Weights  ();
    float regulL2Weights  ();
    int   numberOfWeights ();
    void  displayWeights  ();
    float getWeight       (int, int, int);
    int   setWeight       (const int, const int, const int, const float);
    void  statWeights     ();
    float meanWeights     ();
    float stdevWeights    (const float);

// DS
    using DS_t  = int32_t;
    using uDS_t = uint32_t;
    void  DSrun         (MLMatrix<float>, MLMatrix<float>, int, int, float);
    MLMatrix<DS_t> DSforward (MLMatrix<DS_t>, bool);
    uDS_t DSerror       (MLMatrix<DS_t>, MLMatrix<DS_t>) const;
    void  DSbackward    (const MLMatrix<DS_t>, const MLMatrix<DS_t>, const int);
    void  DSinitWeights (const DS_t, const DS_t);
    void  DSupdate      (const int);

// Misc functions
    int   size            () const;
    void  displayNetwork  ();
    void  netSave         (const char* const);
    bool  netLoad         (const char* const);

// Pruning
    bool     pruneAll     ();
    uint16_t pruneInactive();
    uint16_t pruneLowAct  ();
    uint16_t pruneTopK    (const MLMatrix<float>, const MLMatrix<float>);
    void     removeNeuron (const int, const int);
    void     addNeuron    (const int);

  private:

// Initial values of parameters
    float _eta       = 0.1f;
    float _eta0      = _eta;
    float _momentum  = 0.5f;
    float _gain      = 1.0f;
    float _trainTest = 0.8f;

    float _currError = 100.0f;
    float _etaMin    = 0.01f;
    float _minError  = 1000000.0f;
    float _wmin      = -0.5f;
    float _wmax      =  0.5f;
    float _logLRmax  = -1.0f;
    float _logLRmin  = -4.0f;
    float _minAlpha  = 0.1f;
    float _maxAlpha  = 1.5f;
    float _minMom    = 0.5f;
    float _maxMom    = 0.99f;
    float _minGain   = 0.5f;
    float _maxGain   = 2.0f;
    float _minBS     = 0.05f;
    float _maxBS     = 0.2f;

    float _lambdaRegulL1  = 0.1f;
    float _lambdaRegulL2  = 0.1f;
    float _gradScale      = 1.0f;
    float _gradClipValue  = 0.75f;
    float _zeroThreshold  = 0.15f;
    float _pruningThreshold = 0.85f;
    float _dropout_prob   = 0.1f;

    float rTrain = 4.0f / 6.0f;
    float rValid = 1.0f / 6.0f;
    float rTest  = 1.0f / 6.0f;
    uint8_t _norm    = 0;
    uint8_t _nDots   = 0;
    float _topKpcent = 0.50f; // not lower than 0.4
    uint8_t _minPerLayer = 3;

// Private variables
    uint8_t _nLayers, _nInputs, _nClasses;
    uint8_t _neurons[MAX_LAYERS];
    uint8_t _activations[MAX_LAYERS];
    uint8_t _verbose;
    uint8_t _cost = 0;
    uint16_t _maxEpochs;
    uint16_t _nTrain, _nValid, _nTest;
    uint16_t _nData, _batchSize, _lastBestEpoch = 0;
    float _stopError, _prevMinError, _validError;
    float _prevError, _firstError;
    bool _bestEta      = false;
    bool _firstRun     = true;
    bool _datasetSplit = false;
    bool _enSoftmax    = false;

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
    std::vector<MLMatrix<float> > Weights2;
    std::vector<MLMatrix<float> > Biases;
    std::vector<MLMatrix<float> > dWeights;
    std::vector<MLMatrix<float> > dWeights2;
    std::vector<MLMatrix<float> > dBiases;
    std::vector<MLMatrix<float> > dWeightsOld;
    std::vector<MLMatrix<float> > dWeightsOld2;
    std::vector<MLMatrix<float> > dBiasesOld;
    std::vector<MLMatrix<float> > _a;
    std::vector<MLMatrix<uint8_t> > _dropoutMasks;
    std::vector <std::vector<uint16_t> > b_topK;

// DeepShift 
    std::vector<MLMatrix<DS_t> > _aDS;
    std::vector<MLMatrix<DS_t> > DS_Biases;
    std::vector<MLMatrix<DS_t> > DS_dBiases;
    std::vector<MLMatrix<DS_t> > DS_dBiasesOld;
    std::vector<MLMatrix<DS_t> > P;
    std::vector<MLMatrix<DS_t> > S;
    std::vector<MLMatrix<DS_t> > P2;
    std::vector<MLMatrix<DS_t> > S2;
    std::vector<MLMatrix<DS_t> > dP;
    std::vector<MLMatrix<DS_t> > dS;
    std::vector<MLMatrix<DS_t> > dP2;
    std::vector<MLMatrix<DS_t> > dS2;
    std::vector<MLMatrix<DS_t> > dPOld;
    std::vector<MLMatrix<DS_t> > dSOld;
    std::vector<MLMatrix<DS_t> > dP2Old;
    std::vector<MLMatrix<DS_t> > dS2Old;
    uint8_t _shiftP = 0;

// Saved values (save / restore network)
    std::vector<MLMatrix<float> > Weights_save;
    std::vector<MLMatrix<float> > Weights2_save;
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
    float Swish (const float);
    float dSwish (const float);
    MLMatrix<float> SoftMax (const MLMatrix<float>);

// Private functions
    MLMatrix<float> Mse (MLMatrix<float>, MLMatrix<float>);
    MLMatrix<float> LogLikelihood(MLMatrix<float>);
    MLMatrix<float> activation (MLMatrix<float>, const uint8_t);
    MLMatrix<float> dActivation (MLMatrix<float>, const uint8_t);

    float CrossEntropy (const MLMatrix<float>, const MLMatrix<float>);
    void  heuristics (int, int, bool);
    // void  searchEta (MLMatrix<float>, MLMatrix<float>, float, std::vector<MLMatrix<float> >, std::vector<MLMatrix<float> >, std::vector<MLMatrix<float> >, std::vector<MLMatrix<float> >);
    void  searchEta (MLMatrix<float>, MLMatrix<float>, float);
    int   readIntFile (File);
    float readFloatFile (File);
    void  initWeights (float = -0.5f, float = 0.5f);
    void  topKCount (float, int, MLMatrix<float>&);

    // typedef float (MLP::*Act) (const float);
    // Act Activation[3] = {&MLP::ReLu, &MLP::Sigmoid, &MLP::Tanh};
    // Act dActivation[3] = {&MLP::dReLu, &MLP::dSigmoid, &MLP::dTanh};
    char ActivNames[11][10] = {"RELU", "SIGMOID", "TANH", "SOFTMAX",
                              "SIGMOID2", "IDENTITY", "LEAKYRELU", 
                              "ELU", "SELU", "RELU6", "SWISH"};

// Booleans for the heuristics
    uint32_t _heuristics     = 131157; // default value for small code
    bool     _initialize     = true;   // in default
    bool     _changeWeights  = false;
    bool     _mutateWeights  = false;  // in default
    bool     _changeLRlin    = false;
    bool     _changeLRlog    = false;  // in default
    bool     _changeGain     = false;
    bool     _changeMom      = false;  // in default
    bool     _shuffleDataset = false;
    bool     _zeroWeights    = false;
    bool     _stopTotalError = false;
    bool     _selectWeights  = false;
    bool     _xavier         = false;
    bool     _regulL1        = false;
    bool     _regulL2        = false;
    bool     _labelSmoothing = false;
    bool     _gradClip       = false;
    bool     _gradScaling    = false;  // in default
    bool     _varMom         = false;
    bool     _dataSubset     = false;
    bool     _prune_topk     = false;
    bool     _prune_neurons  = false;
    bool     _prune_train    = false;
    bool     _dropout        = false;
    bool     _changeBSize    = false;
    bool     _quadLayers     = false;
    bool     _deepShift      = false;
};

inline float halfSquare (const float x) { return 0.5f * pow(x, 2); }
inline float minusLog   (const float x) { return -log(x + 1.0e-15); }
inline float rand01 () { return float(random(10000)) / 10000.0f; }

#endif
