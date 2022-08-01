#include "MMLP.h"
#define MIN_float      -HUGE_VAL
#define MAX_float      +HUGE_VAL

#define DEBUG 0


/*
  Constructor, arguments are:
    number of layers
    array of number of neurons
    verbose level (0= silent, 1= intermediary, 2= very talkative, 3 = even more)
    'skip' enables (true) or disables (false) to add to each layer inputs from layer l-2
*/
MLP::MLP(const int *neurons, const int nLayers, const int verbose)
{
	_verbose = verbose;
	if (_verbose > 0) Serial.printf ("\n\nVerbose level : %d\n", _verbose);
	_nLayers = nLayers;
	if (_nLayers > MAX_LAYERS) Serial.printf ("Too many layers (%d), maximum is: %d\n",_nLayers, MAX_LAYERS);
	for (size_t i = 0; i < _nLayers; ++i) _neurons[i] = neurons[i];
	_nInputs = _neurons[0];
	_nClasses = _neurons[_nLayers - 1];

	if (_verbose > 0) {
		Serial.println("Creating network:");
		Serial.printf("%d layers\n",_nLayers);
		Serial.printf("\tInput layer: %d neurons\n", _nInputs);
		for (size_t i = 1; i < _nLayers -1; ++i)
			Serial.printf("\tHidden layer %d: %d neurons\n",i,_neurons[i]);
		Serial.printf("\tOutput layer: %d neurons", _nClasses);
	}
}

MLP::~MLP () { }

/************************************

			Initializations 

*************************************/
void MLP::randomWeights (float min, float max) 
{
	_wmin = min;
	_wmax = max;
	if (_verbose > 1) Serial.println("Initializing random weights");
	initWeights ();
}

// Initialize weights & biases : random uniform or Xavier
void MLP::initWeights (float vmin, float vmax) 
{
	Weights.clear();
	Biases.clear();
	if (_quadLayers) Weights2.clear();

	for (size_t k = 0; k < _nLayers - 1; ++k) {
		MLMatrix<float> W(_neurons[k + 1], _neurons[k], vmin, vmax);
		MLMatrix<float> B(_neurons[k + 1], 1, vmin, vmax);

		if (_xavier) { // Normal Xavier initialization
				float mean = 0.0f;
				float std_dev = sqrt(2.0f / (_neurons[k + 1] + _neurons[k]));
				W.randomNormal(mean, std_dev);
				std_dev = sqrt(2.0f / (_neurons[k + 1]));
				B.randomNormal(mean, std_dev);
		}

		Weights.push_back(W);
		Biases.push_back(B);

		if (_quadLayers) {
			MLMatrix<float> W2(_neurons[k + 1], _neurons[k], vmin, vmax);
			if (_xavier) {
				float mean = 0.0f;
				float std_dev = sqrt(2.0f / (_neurons[k + 1] + _neurons[k]));
				W2.randomNormal(mean, std_dev);
			}
			Weights2.push_back(W2);
		}

		if (_verbose > 2) {
			W.print();
			if (_quadLayers) {
				MLMatrix<float> W2 = Weights2.back();
				W2.print();
			}
			B.print();
		}
	}
}

void MLP::setActivations (const int *activations)
{
	for (size_t i = 0; i < _nLayers - 1; ++i) {
		_activations[i] = activations[i];
		if (i != _nLayers - 2 && _activations[i] == SOFTMAX) {
			Serial.printf ("Layer %d cannot use SOFTMAX activation");
			while (1);
		}		
	}

	if (_verbose > 0) {
		Serial.println ("Setting activations:");
		for (size_t i = 0; i < _nLayers - 1; ++i)
			Serial.printf("\tLayer %d: activation %s\n", i, ActivNames[_activations[i]]);
	}

	if (_activations[_nLayers - 2] == SOFTMAX) {
		if (_neurons[_nLayers - 1] > 1) _enSoftmax = true;
		else {
			Serial.println ("Unable to use SOFTMAX activation with only 1 class.");
			while (1);
		}
	}
}

void MLP::setCost (const uint8_t cost)
{
	if ((cost < MSE) || (cost > LOGLIKELIHOOD)) {
		Serial.println("setCost error: cost must be equal to 0, 1 or 2");
			while (1);
	}

	if (cost == CROSSENTROPY && _activations[_nLayers - 2] == RELU)
		Serial.println ("Warning: Cross Entropy used with RELU activation in last layer!");
	if (cost == LOGLIKELIHOOD && _activations[_nLayers - 2] == RELU)
		Serial.println ("Warning: Log Likelihood used with RELU activation in last layer!");

	_cost = cost;
}

// Set the values of learning rate (called eta) and momentum
void MLP::setHyper (const float eta, const float momentum)
{
	_eta = eta;
	_eta0 = eta;
	_logLRmax = log10(eta);
	_momentum = momentum;

	if (_verbose > 0)	{
		Serial.println ("Setting hyperparameters:");
		Serial.printf("- Learning rate = %f\n- Momentum      = %f\n", _eta, _momentum);
	}
}

// Set the ratio of train and test in the dataset (default 66.6% train, 16.6% test, 16.6% validation)
void MLP::setTrainTest (float rTrain, float rValid, float rTest)
{
	float sum = rTrain + rValid + rTest;
	if (sum != 1) {
		rTrain /= sum;
		rValid /= sum;
		rTest  /= sum;
		Serial.printf ("Sum of ratios is not 100%% (%.1f%%): setting ratios to %.2f, %.2f, %.2f\n", sum*100, rTrain, rValid, rTest);
	}
	_nTrain = _nData * rTrain;
	_nValid = _nData * rValid;
	_nTest  = _nData * rTest;
	Serial.printf ("Dataset split in: %d train + %d validation + %d test data\n", _nTrain, _nValid, _nTest);
	_datasetSplit = true;
}

/************************************

			Parameters and heuristics 

*************************************/

void MLP::setHeuristics (uint32_t heuristics)
{
  _heuristics = heuristics;
  if (_heuristics != 0) {
    _initialize     = _heuristics & H_INIT_OPTIM; // 0x0000001
    _changeWeights  = _heuristics & H_CHAN_WEIGH; // 0x0000002
    _mutateWeights  = _heuristics & H_MUTA_WEIGH; // 0x0000004
    _changeLRlin    = _heuristics & H_CHAN_LRLIN; // 0x0000008
    _changeLRlog    = _heuristics & H_CHAN_LRLOG; // 0x0000010
    _changeGain     = _heuristics & H_CHAN_SGAIN; // 0x0000020
    _changeMom      = _heuristics & H_CHAN_MOMEN; // 0x0000040
    _shuffleDataset = _heuristics & H_SHUF_DATAS; // 0x0000080
    _zeroWeights    = _heuristics & H_ZERO_WEIGH; // 0x0000100
    _stopTotalError = _heuristics & H_STOP_TOTER; // 0x0000200
    _xavier         = (_initialize) ? (_heuristics & H_INIT_XAVIE) : 0; // 0x000800 (not if reading from file)
    _selectWeights  = (!_initialize || _xavier) ? 0 : (_heuristics & H_SELE_WEIGH); // 0x000400 (not if reading from file)
    _regulL1        = _heuristics & H_REG1_WEIGH; // 0x0001000
    _regulL2        = _heuristics & H_REG2_WEIGH; // 0x0002000
    _bestEta        = _heuristics & H_BEST_ETA  ; // 0x0004000
    _labelSmoothing = _heuristics & H_LABL_SMOOT; // 0x0008000
    _gradClip       = _heuristics & H_GRAD_CLIP ; // 0x0010000
    _gradScaling    = _heuristics & H_GRAD_SCALE; // 0x0020000
    _varMom         = _heuristics & H_CHAN_MOLIN; // 0x0040000
    _dataSubset     = _heuristics & H_DATA_SUBSE; // 0x0080000
    _prune_topk     = _heuristics & H_TOPK_PRUNE; // 0x0100000
    _prune_neurons  = _heuristics & H_TEST_PRUNE; // 0x0200000
    _prune_train    = _heuristics & H_TRAI_PRUNE; // 0x0400000
    _dropout        = _heuristics & H_DROP_OUT  ; // 0x0800000
    _changeBSize    = _heuristics & H_CHAN_BATSZ; // 0x1000000
    _quadLayers     = _heuristics & H_QUAD_LAYER; // 0x2000000
    _deepShift      = _heuristics & H_DEEP_SHIFT; // 0x4000000
  }
}

void MLP::displayHeuristics ()
{
  Serial.println("---------------------------");
  Serial.println("Heuristics parameters:");
  if (_initialize && !_xavier) Serial.println ("- Init with random weights");
  if (_xavier)          Serial.println ("- Init weights with Xavier method");
  if (_selectWeights)   Serial.println ("- Select best weights at init");
  if (_dataSubset)      Serial.println ("- Begin training on a subset of the dataset");
  if (_shuffleDataset)  Serial.println ("- Shuffle dataset if needed");
  if (_changeWeights)   Serial.println ("- Random weights if needed");
  if (_mutateWeights)   Serial.println ("- Slightly change weights if needed");
  if (_changeLRlin)     Serial.printf  ("- Variable learning rate (linear scale from %.3f to %.3f)\n", _eta, _etaMin);
  if (_changeLRlog)     Serial.printf  ("- Variable learning rate (log scale from 10^%.3f to 10^%.3f)\n", _logLRmax, _logLRmin);
  if (_changeGain)      Serial.printf  ("- Random variable Sigmoid gain between %.3f and %.3f\n", _minGain, _maxGain);
  if (_changeMom)       Serial.printf  ("- Random variable momentum between %.3f and %.3f\n", _minAlpha, _maxAlpha);
  if (_varMom)          Serial.printf  ("- Quadratic variable momentum from %.3f to %.3f\n", _maxMom, _minMom);
  if (_zeroWeights)     Serial.printf  ("- Force weights less than %.3f to zero\n", _zeroThreshold);
  if (_stopTotalError)  Serial.println ("- Stop optimization if train + validation errors under threshold");
  if (_regulL1)         Serial.printf  ("- Use L1 weight regularization (lambda = %.3f)\n", _lambdaRegulL1);
  if (_regulL2)         Serial.printf  ("- Use L2 weight regularization (lambda = %.3f)\n", _lambdaRegulL2);
  if (_bestEta)					Serial.println ("- Search for best learning rate at each epoch (experimental)");
  if (_labelSmoothing)  Serial.println ("- Label smoothing ON");
  if (_gradClip)        Serial.printf  ("- Gradient clipping (clip value over %.3f)\n", _gradClipValue);
  if (_gradScaling)     Serial.printf  ("- Use gradient scaling (Norm to %.3f)\n", _gradScale);
  if (_quadLayers)      Serial.println ("- Use quadratic layers");
  if (_quadLayers) 		  { _dropout = false;}
  if (_prune_topk)			Serial.println ("- TopK pruning ON");
  if (_prune_train)     _prune_neurons = true;
  if (_prune_train)     Serial.println ("- Prune inactive or low activity neurons during training phase");
  if (_prune_neurons)   Serial.println ("- Prune inactive or low activity neurons at test phase");
  if (_dropout)	        Serial.printf  ("- Dropout, with probability %.2f\n", _dropout_prob);
  if (_changeBSize)     Serial.println ("- Change batch size during training");
  if (_deepShift)       Serial.println ("- Use Deep Shift quantization algorithm  (experimental...)");
  // if (_parallelRun)     Serial.println ("- Compute using both processors");
  // if (_enableSkip)      Serial.println ("- Layer skip enabled (ResNet like)");
  Serial.println("---------------------------");
}

// Set the value of '_initialize'
void MLP::setHeurInitialize (const bool val) {
  _initialize = val;
  if (!_initialize) _selectWeights = false;
  if (!_initialize) _xavier = false;
}

void MLP::setVerbose (const uint8_t verbose) {
  /*
     verbose levels:
     0: silent
     1: show progression
     2: details of all training steps
     3: 2 + content of matrices, dataset csv file, etc.
  */
  _verbose = verbose;
}

void MLP::setEpochs (const int epochs) { _maxEpochs = epochs; }
void MLP::setBatchSize (const int batchSize) {
  _batchSize = batchSize;
  // if (_enablePS) _batchSize = 1;
}
void MLP::setMomentum (const float momentum) {
  _momentum = momentum;
  _momentum_save = momentum;
}
void MLP::setEta (const float eta) {
  _eta = eta;
  _eta_save = eta;
}

// Set the variation range for the learning rate (either lin or log scale)
void MLP::setEtaRange (const float from, const float to)
{
	if (_changeLRlog) { // input log values (e.g. -2, -4)
		_logLRmax = from;
		_logLRmin = to;
	} else {            // input linear valuers (e.g. 0.01, 0.0001)
		_eta0 = from;
		_etaMin = to;
	}
}

void MLP::setMomRange  (const float from, const float to)
{
	_minMom   = from;
	_maxMom   = to;
	_minAlpha = from;
	_maxAlpha = to;
}

// Set the gain of Sigmoid (to change the slope at origin)
void MLP::setGain (const float gain) {
  _gain = gain;
  _gain_save = gain;
}

// Return values of sme hyper parameters
int   MLP::getEpochs () 		{ return _maxEpochs; }
int   MLP::getBatchSize () 	{ return _batchSize; }
float MLP::getMomentum () 	{ return _momentum; }
float MLP::getEta () 				{ return _eta; }
float MLP::getGain () 			{ return _gain; }
int   MLP::getNeuronNumbers (int layer)
{
	if (layer >= _nLayers) return 0;
  return _neurons[layer];
}

float MLP::getWeight (int layer, int lower, int upper) {
  if (layer >= _nLayers) return 0;
  if (upper > _neurons[layer]) return 0;
  if (lower > _neurons[layer - 1]) return 0;
  return Weights[layer](upper, lower);
}

int MLP::setWeight (const int layer, const int upper, const int lower, const float val) {
  if (layer >= _nLayers) return 0;
  if (upper > _neurons[layer]) return 0;
  if (lower > _neurons[layer - 1]) return 0;
  Weights[layer](upper, lower) = val;
  return 1;
}

void MLP::setHeurShuffleDataset (const bool val) { _shuffleDataset = val; }
void MLP::setHeurZeroWeights (const bool val, const float zeroThreshold) {
  _zeroWeights   = val;
  _zeroThreshold = zeroThreshold;
}
void MLP::setHeurRegulL1 (const bool val, const float lambda) {
  _regulL1 = val;
  _lambdaRegulL1 = lambda;
}
void MLP::setHeurRegulL2 (const bool val, const float lambda) {
  _regulL2 = val;
  _lambdaRegulL2 = lambda;
}

void MLP::setHeurGradScale (const bool val, const float scale) { 
	_gradScaling   = val,
	_gradScale     = scale; 
}
void MLP::setHeurGradClip (const bool val, const float clip) { 
	_gradClip      = val;
	_gradClipValue = clip; 
}

void MLP::setHeurChangeMomentum (const bool val, const float minAlpha, const float maxAlpha) {
  _changeMom      = val;
  _minAlpha       = minAlpha;
  _maxAlpha       = maxAlpha;
}
void MLP::setHeurChangeGain (const bool val, const float minGain, const float maxGain) {
  _changeGain    = val;
  _minGain       = minGain;
  _maxGain       = maxGain;
}
void MLP::setHeurPruning (const bool val, const float threshold){
	_prune_neurons = val;
	_pruningThreshold = threshold;
}

void MLP::setHeurTopK (const bool val, const float percent)
{
	_prune_topk = val;
	_topKpcent  = percent; // in [0, 1]
}

void MLP::setBSizeRange (float minBS, float maxBS)
{
	if (minBS > 0.9) minBS = 0.9;
	if (maxBS > 0.9) maxBS = 0.9;
	_minBS = minBS;
	_maxBS = maxBS;
}


/************************************

			Activations 

*************************************/
float MLP::ReLu (const float x) { return (x < 0) ? 0 : x; }
float MLP::dReLu (const float x) { return (x < 0) ? 0 : 1; } // derivative

float MLP::ReLu6 (const float x) { return (x < 0) ? 0 : (x > 6) ? 6 : x; }
float MLP::dReLu6 (const float x) { return (x < 0 || x > 6) ? 0 : 1; }

float MLP::Sigmoid (const float x) { 
	if (_gain * x < -5.0f) return 0.0f;
	if (_gain * x >  5.0f) return 1.0f;
	return 1.0f / (1.0f + exp(-_gain * x));
}
float MLP::dSigmoid (const float x) { return _gain * x * (1.0f - x); } // derivative

float MLP::Sigmoid2 (const float x) { 
	if (_gain * x < -5.3f) return -1.0f;
	if (_gain * x >  5.3f) return  1.0f;
	return 1.0f - 2.0f / (1.0f + exp(_gain * x));
}
float MLP::dSigmoid2 (const float x) { return _gain * (1.0f + x) * (1.0f - x) / 2.0f; } // derivative

float MLP::Tanh (const float x) { return tanh(x); }
float MLP::dTanh (const float x) { return 1.0f - x * x; } // derivative

float MLP::Id (const float x) { return x; }
float MLP::dId (const float x) { return 1.0f; }

float MLP::SeLu (const float x) { 
	return (x > 0) ? 1.0507f * x : 1.0507f * 1.6732632f * (exp(x) - 1.0f); }
	//               lambda        lambda      alpha
float MLP::dSeLu (const float x) { 
	return (x > 0) ? 1.0507f : 1.0507f * 1.6732632f + x; } // x + lambda * alpha

float MLP::LeakyReLu (const float x) { return (x > 0) ? x : x * 0.1f; }
float MLP::dLeakyReLu (const float x) { return (x > 0) ? 1.0f : 0.1f; }

float MLP::ELu (const float x) { 
	float _alphaELU = 1.0f;
	return (x > 0) ? x : _alphaELU * (exp(x) - 1);
}
float MLP::dELu (const float x) { 
	float _alphaELU = 1.0f;
	return (x > 0) ? 1.0f : _alphaELU + x;
}
float MLP::Swish (const float x) {
	float beta = 4.0f / 30.0f;
	return x * Sigmoid(beta * x);
}
float MLP::dSwish (const float x) {
	float beta = 4.0f / 30.0f;
	float y = Sigmoid(beta * x);
	return beta * x * y + y * (1.0f - beta * x * y) ;
}

MLMatrix<float> MLP::SoftMax(const MLMatrix<float> x) {
	MLMatrix<float> S(x, 0.0f);
	float maxX = x.max();
	float sum = 0.0f;
	for (size_t i = 0; i < x.get_rows(); ++i) sum += exp(x(i, 0) - maxX);
	for (size_t i = 0; i < x.get_rows(); ++i) S(i, 0) = exp(x(i, 0) - maxX) / sum;
	return S;
}

/************************************

			Costs

*************************************/
// yhat: output of the forward pass, y: ground truth
float MLP::CrossEntropy(const MLMatrix<float> yhat, const MLMatrix<float> y) {
	float sum = 0.0f;
	for (size_t i = 0; i < yhat.get_rows(); ++i) {
		sum -=         y(i, 0)  * log(yhat(i, 0) + 1.0e-15);
		sum -= (1.0f - y(i, 0)) * log(1.0f - yhat(i, 0) + 1.0e-15);
	}
	return - sum / y.get_rows();
}

// yhat: output of the forward pass, y: ground truth
MLMatrix<float> MLP::Mse(MLMatrix<float> yhat, MLMatrix<float> y) {
	MLMatrix<float> Err = y - yhat;
	Err.applySelf(&halfSquare);
	return Err;
}

// yhat: output of the forward pass
MLMatrix<float> MLP::LogLikelihood(MLMatrix<float> yhat) {
	MLMatrix<float> Err = yhat;
	Err.applySelf(&minusLog);
	return Err;
}


/************************************

			Utilities 

*************************************/
int MLP::size() const
{
	Serial.printf ("Network has %d layers:\n", _nLayers);
	int nbSynapses = 0;
	for (size_t i = 0; i < _nLayers; ++i) {
		if (i==0) {
			if (_verbose > 0) Serial.printf("\tLayer %d: %d neurons\n",i, _neurons[i]);
		} else {
			int nWeights = _neurons[i] * _neurons[i-1]; // weights
			if (_quadLayers) nWeights *= 2;
			int nBiases= _neurons[i]; // biases
			nbSynapses += (nWeights + nBiases);
			if (_verbose > 0) Serial.printf("\tLayer %d: %2d neurons, activation %s, %d weights, %d biases\n",
												i, _neurons[i], ActivNames[_activations[i-1]], nWeights, nBiases);			
		}
	}
	Serial.printf("\tTotal number of synapses: %d (i.e. weights + biases)\n", nbSynapses);
	return nbSynapses;
}

// Display the parameters of the network
void MLP::displayNetwork() 
{
	restoreWeights();
	Serial.println("\n---------------------------");
	int nbSynapses = size();
	Serial.printf("Average L1 norm of synapses: %.3f\n",regulL1Weights() / nbSynapses);
	Serial.printf("Average L2 norm of synapses: %.3f\n",regulL2Weights() / nbSynapses);
	float mean = meanWeights();
	Serial.printf("Average value of synapses:   %.3f\n", mean);
	Serial.printf("Standard dev. of synapses:   %.3f\n", stdevWeights(mean));
	statWeights();
	Serial.printf("\nFinal learning rate: %.3f\n", _eta);
	Serial.printf("Final Sigmoid gain : %.3f\n", _gain);
	Serial.printf("Final momentum     : %.3f\n", _momentum);
	Serial.println("---------------------------");
}

// Compute statistics on weights
void MLP::statWeights()
{
	int great0 = 0;
	int less0  = 0;
	int less1  = 0;
	int less2  = 0;
	int less3  = 0;
	int less4  = 0;
	for (size_t k = 0; k < _nLayers - 1; ++k) {
		// Weights
		MLMatrix<float> W = Weights[k];
		for (size_t i = 0; i < _neurons[k+1]; ++i) 
			for (size_t j = 0; j < _neurons[k]; ++j) {
				float ww = abs(W(i,j));
				if (ww >= 1.0)                   ++great0;
				if (ww < 1.0   && ww >= 0.1)     ++less0;
				if (ww < 0.1   && ww >= 0.01)    ++less1;
				if (ww < 0.01  && ww >= 0.001)   ++less2;
				if (ww < 0.001 && ww >= 0.0001)  ++less3;
				if (ww < 0.0001)                 ++less4;
			}
		if (_quadLayers) {
			MLMatrix<float> W2 = Weights2[k];
			for (size_t i = 0; i < _neurons[k+1]; ++i) 
				for (size_t j = 0; j < _neurons[k]; ++j) {
					float ww = abs(W2(i,j));
					if (ww >= 1.0)                   ++great0;
					if (ww < 1.0   && ww >= 0.1)     ++less0;
					if (ww < 0.1   && ww >= 0.01)    ++less1;
					if (ww < 0.01  && ww >= 0.001)   ++less2;
					if (ww < 0.001 && ww >= 0.0001)  ++less3;
					if (ww < 0.0001)                 ++less4;
				}
		}
		// Biases
		MLMatrix<float> B = Biases[k];
		for (size_t i = 0; i < _neurons[k+1]; ++i) {
			float ww = abs(B(i,0));
			if (ww >= 1.0)                   ++great0;
			if (ww < 1.0   && ww >= 0.1)     ++less0;
			if (ww < 0.1   && ww >= 0.01)    ++less1;
			if (ww < 0.01  && ww >= 0.001)   ++less2;
			if (ww < 0.001 && ww >= 0.0001)  ++less3;
			if (ww < 0.0001)                 ++less4;
		}
	}
	int nSynapses = numberOfWeights();
	Serial.printf("Ratio of synapses greater than 1:   %5.2f %%\n", 100.0f * great0 / nSynapses);
	Serial.printf("Ratio of synapses less than 1:      %5.2f %%\n", 100.0f * less0  / nSynapses);
	Serial.printf("Ratio of synapses less than 0.1:    %5.2f %%\n", 100.0f * less1  / nSynapses);
	Serial.printf("Ratio of synapses less than 0.01:   %5.2f %%\n", 100.0f * less2  / nSynapses);
	Serial.printf("Ratio of synapses less than 0.001:  %5.2f %%\n", 100.0f * less3  / nSynapses);
	Serial.printf("Ratio of synapses less than 0.0001: %5.2f %% (sparsity)\n", 100.0f * less4 / nSynapses);
}


// Store the weights and parameters for later use
void MLP::saveWeights()
{
	if (_verbose > 1) Serial.println("Saving weights");
	_eta_save = _eta;
	_momentum_save = _momentum;
	_gain_save = _gain;
	Weights_save.clear();
	Biases_save.clear();
	Weights_save = Weights;
	Biases_save = Biases;
	if (_quadLayers) {
		Weights2_save.clear();
		Weights2_save = Weights2;
	}
}

// Store the weights and parameters for later use
void MLP::restoreWeights()
{
	if (_verbose > 1) Serial.println("Restoring weights");
	// _eta = _eta_save;
	// _momentum = _momentum_save;
	// _gain = _gain_save;
	Weights.clear();
	Biases.clear();
	Weights = Weights_save;
	Biases = Biases_save;
	if (_quadLayers) {
		Weights2.clear();
		Weights2 = Weights2_save;
	}
	for (unsigned i = 1; i < _nLayers; ++i) _neurons[i] = Weights[i - 1].get_rows();
	Serial.println("Apres restore"); size();
}

// Loads a network from LITTLEFS file system
bool MLP::netLoad(const char* const path)
{
	File file = LITTLEFS.open(path);
	if (!file || file.isDirectory()) {
		Serial.printf("%s - failed to open file for reading\n", path);
		return false;
	}

	// Load parameters of layers
	if (_verbose > 0) Serial.printf("Loading network from file %s\n", path);
	_nLayers = readIntFile (file);
	if (_verbose > 1) Serial.printf ("%d layers\n", _nLayers);
	for (size_t k = 0; k < _nLayers; ++k) {
		_neurons[k] = readIntFile (file);
		if (k > 0) _activations[k-1] = readIntFile (file);
		if (_verbose > 1) {
			if (k > 0) Serial.printf("Layer %d -> %d neurons, %s\n", k,
		  	_neurons[k], ActivNames[_activations[k-1]]);
		  else Serial.printf("Layer %d -> %d neurons\n", k, _neurons[k]);
		}
	}

	// Load parameters
	_momentum_save = readFloatFile (file);
	_eta_save = readFloatFile (file);
	_gain_save = readFloatFile (file);
	if (_verbose > 1) Serial.printf("Momentum      = %f\n", _momentum_save);
	if (_verbose > 1) Serial.printf("Learning rate = %f\n", _eta_save);
	if (_verbose > 1) Serial.printf("Sigmoid gain  = %f\n", _gain_save);

	// Load layers
	int nW = 0;
	Weights.clear();
	if (_quadLayers) Weights2.clear();
	Biases.clear();
	for (size_t k = 0; k < _nLayers - 1; ++k) {

		if (_quadLayers) {
			MLMatrix<float> W2(_neurons[k + 1], _neurons[k], 0);
			for (size_t i = 0; i < _neurons[k+1]; ++i) {
				for (size_t j = 0; j < _neurons[k]; ++j) {
					W2(i,j) = readFloatFile (file);
					if (_verbose > 2) 
						Serial.printf("Layer %d: loading weight (%d, %d) = %.6f\n", k, i, j, W2(i,j));
					++nW;
				}
			}
			Weights2.push_back(W2);
		}

		MLMatrix<float> W(_neurons[k + 1], _neurons[k], 0);
		MLMatrix<float> B(_neurons[k + 1], 1, 0);
		for (size_t i = 0; i < _neurons[k+1]; ++i) {
			for (size_t j = 0; j < _neurons[k]; ++j) {
				W(i,j) = readFloatFile (file);
				if (_verbose > 2) 
					Serial.printf("Layer %d: loading weight (%d, %d) = %.6f\n", k, i, j, W(i,j));
				++nW;
			}
		}
		for (size_t i = 0; i < _neurons[k+1]; ++i) {
			B(i,0) = readFloatFile (file);
			if (_verbose > 2) 
				Serial.printf("Layer %d: loading bias (%d) = %.6f\n", k, i, B(i,0));
			++nW;
		}
		Weights.push_back(W);
		Biases.push_back(B);
	}
	if (_verbose > 1) Serial.printf ("Loaded %d weights\n", nW);
	file.close();
	return true;
}

// Saves a network to LITTLEFS file system
void MLP::netSave(const char* const path)
{
	restoreWeights();
	File file = LITTLEFS.open(path, FILE_WRITE);
	if (!file) {
		Serial.printf("%s - failed to open file for writing\n", path);
		return;
	}

	// Save parameters of layers
	if (_verbose > 0) Serial.printf("Saving network in file %s\n", path);
	file.printf("%d\n", _nLayers);
	if (_verbose > 1) Serial.printf ("%d layers\n", _nLayers);
	for (size_t k = 0; k < _nLayers; ++k) {
		file.printf("%d\n", _neurons[k]);
		if (k > 0) file.printf("%d\n", _activations[k-1]);
		if (_verbose > 1) {
			if (k > 0) Serial.printf("Layer %d -> %d neurons, %s\n", k,
		  	_neurons[k], ActivNames[_activations[k-1]]);
		  else Serial.printf("Layer %d -> %d neurons\n", k, _neurons[k]);
		}
	}

	// Save parameters
	file.printf("%f\n%f\n%f\n", _momentum_save, _eta_save, _gain_save);
	if (_verbose > 1) Serial.printf("Momentum = %f\n", _momentum_save);
	if (_verbose > 1) Serial.printf("Learning rate = %f\n", _eta_save);
	if (_verbose > 1) Serial.printf("Gain = %f\n", _gain_save);

	// Save layers
	int nW = 0;
	for (size_t k = 0; k < _nLayers - 1; ++k) {
		// Weights

		if (_quadLayers) {
			MLMatrix<float> W2 = Weights2[k];
			for (size_t i = 0; i < _neurons[k+1]; ++i) {
				for (size_t j = 0; j < _neurons[k]; ++j) {
					file.printf("%.6f\n", W2(i,j));
					if (_verbose > 2) 
						Serial.printf("Layer %d: saving weight (%d, %d) = %.6f\n", k, i, j, W2(i,j));
					++nW;
				}
			}
		}

		MLMatrix<float> W = Weights[k];
		for (size_t i = 0; i < _neurons[k+1]; ++i) {
			for (size_t j = 0; j < _neurons[k]; ++j) {
				file.printf("%.6f\n", W(i,j));
				if (_verbose > 2) 
					Serial.printf("Layer %d: saving weight (%d, %d) = %.6f\n", k, i, j, W(i,j));
				++nW;
			}
		}
		// Biases
		MLMatrix<float> B = Biases[k];
		for (size_t i = 0; i < _neurons[k+1]; ++i) {
			file.printf("%.6f\n", B(i,0));
			if (_verbose > 2) 
				Serial.printf("Layer %d: saving bias (%d) = %.6f\n", k, i, B(i,0));
			++nW;
		}
	}
	if (_verbose > 1) Serial.printf ("Saved %d weights\n", nW);
	file.close();
}


// Reads a positive integer in a file
int MLP::readIntFile (File file) {
  char buffer[15];
  uint8_t i = 0;
  while (file.available()) {
    char c = file.read();
    buffer[i++] = c;
    if (c == 10) return atoi(buffer); // CR
  }
}

// Reads a float from a file
float MLP::readFloatFile (File file) {
  char buffer[20];
  int i = 0;
  while (file.available()) {
    char c = file.read();
    buffer[i++] = c;
    if (c == 10) return atof(buffer); // CR
  }
}

/* Estimate the duration of the complete training in ms
   The estimation is not very precise as the training time
   may depend of other factors such as the verbose level
   and the training may stop before the last epoch is reached
*/
uint32_t MLP::estimateDuration (int maxEpochs)
{
	saveWeights();
  unsigned long chrono = millis();

	MLMatrix<float> x(_nInputs, 1, 0.0f);  // Input array
	MLMatrix<float> y(_nClasses, 1, 0.0f); // ground truth
	// Forward pass
	MLMatrix<float> yhat = forward (x, false);
	// Compute error
	float err = error (y, yhat);
	// Backward pass
	backward (yhat, y, 0);
	// Update weights
	dWeightsOld = dWeights;
	if (_quadLayers) dWeightsOld2 = dWeights2;
	dBiasesOld = Biases;
	update (_batchSize);

  chrono = millis() - chrono;
  restoreWeights();

	uint32_t dur = (chrono * maxEpochs * _nData) / 2;
  return dur;
}

// Weights regularization L1
float MLP::regulL1Weights()
{
	float sum = 0.0f;
	for (size_t k = 0; k < _nLayers - 1; ++k) {
		// Weights
		MLMatrix<float> W = Weights[k];
		for (size_t i = 0; i < _neurons[k+1]; ++i) 
			for (size_t j = 0; j < _neurons[k]; ++j) 
				sum += abs(W(i,j));
		if (_quadLayers) {
			MLMatrix<float> W2 = Weights2[k];
			for (size_t i = 0; i < _neurons[k+1]; ++i) 
				for (size_t j = 0; j < _neurons[k]; ++j) 
					sum += abs(W2(i,j));
		}
		// Biases
		MLMatrix<float> B = Biases[k];
		for (size_t i = 0; i < _neurons[k+1]; ++i) 
			sum += abs(B(i,0));
	}
	return sum;
}

// Weights regularization L2
float MLP::regulL2Weights()
{
	float sum = 0.0f;
	for (size_t k = 0; k < _nLayers - 1; ++k) {
		// Weights
		MLMatrix<float> W = Weights[k];
		for (size_t i = 0; i < _neurons[k+1]; ++i) 
			for (size_t j = 0; j < _neurons[k]; ++j) 
				sum += W(i,j) * W(i,j);
		if (_quadLayers) {
			MLMatrix<float> W2 = Weights2[k];
			for (size_t i = 0; i < _neurons[k+1]; ++i) 
				for (size_t j = 0; j < _neurons[k]; ++j) 
					sum += W2(i,j) * W2(i,j);
		}
		// Biases
		MLMatrix<float> B = Biases[k];
		for (size_t i = 0; i < _neurons[k+1]; ++i) 
			sum += B(i,0) * B(i,0);
	}
	return sum / 2.0f;
}

// Returns the number of weights & biases (aka synapses)
int MLP::numberOfWeights()
{
	int nbSynapses = 0;
	for (size_t i = 1; i < _nLayers; ++i) {
		int nWeights = _neurons[i] * _neurons[i-1]; // weights
		if (_quadLayers) nWeights *= 2;
		int nBiases= _neurons[i]; // biases
		nbSynapses += (nWeights + nBiases);
	}
	return nbSynapses;
}

// Compute the mean values of all weights and biases
float MLP::meanWeights()
{
	float sum = 0.0f;
	for (size_t k = 0; k < _nLayers - 1; ++k) {
		// Weights
		MLMatrix<float> W = Weights[k];
		for (size_t i = 0; i < _neurons[k+1]; ++i) 
			for (size_t j = 0; j < _neurons[k]; ++j) 
				sum += W(i,j);
		if (_quadLayers) {
			MLMatrix<float> W2 = Weights2[k];
			for (size_t i = 0; i < _neurons[k+1]; ++i) 
				for (size_t j = 0; j < _neurons[k]; ++j) 
					sum += W2(i,j);
		}
		// Biases
		MLMatrix<float> B = Biases[k];
		for (size_t i = 0; i < _neurons[k+1]; ++i) 
			sum += B(i,0);
	}
	return sum / numberOfWeights();
}

// Compute the standard deviation of all weights and biases
float MLP::stdevWeights (const float mean)
{
  float stdev = 0.0f;
	for (size_t k = 0; k < _nLayers - 1; ++k) {
		// Weights
		MLMatrix<float> W = Weights[k];
		for (size_t i = 0; i < _neurons[k+1]; ++i) 
			for (size_t j = 0; j < _neurons[k]; ++j) 
				stdev += pow(W(i,j) - mean, 2);
		if (_quadLayers) {
			MLMatrix<float> W2 = Weights2[k];
			for (size_t i = 0; i < _neurons[k+1]; ++i) 
				for (size_t j = 0; j < _neurons[k]; ++j) 
					stdev += pow(W2(i,j) - mean, 2);
		}
		// Biases
		MLMatrix<float> B = Biases[k];
		for (size_t i = 0; i < _neurons[k+1]; ++i) 
			stdev += pow(B(i,0) - mean, 2);
	}
  stdev /= numberOfWeights();
  stdev = sqrt(stdev);
  return stdev;
}

void MLP::displayWeights()
{
	Serial.println("Displaying weights");
	for (size_t k = 0; k < _nLayers - 1; ++k) {
		Serial.printf("Layer %d:\n", k);
		Serial.println("Weights:");
	// Weights
		if (_quadLayers) {
			MLMatrix<float> W2 = Weights2[k];
			W2.print();
		}
		MLMatrix<float> W = Weights[k];
		W.print();
		// Biases
		Serial.println("Biases:");
		MLMatrix<float> B = Biases[k];
		B.print();
	}
}


/************************************

		Dataset functions

		A dataset is a collection of input and output data samples.
		One sample is made of _nInputs input values and _nClasses output values
		(_nInputs is the number of input neurons, _nClasses of output neurons)

		The number of samples is _nData

		A dataset is made of 2 vectors of _nData vectors:
		- input vector  : _nData vectors of _nInputs values
		- output vector : _nData vectors of _nClasses values

		The method createDataset takes these 2 vectors as arguments, you need to create them
		by yourself.
		In the simplified case of one input neuron and one output neuron, you can use the
		methods createDatasetFromArray or createDatasetFromVector

*************************************/
// Process the dataset (number of data, min and max values)
// First case: dataset already a vector of vectors
void MLP::createDataset (MLMatrix<float> x0, MLMatrix<float> &y0, const int nData)
{
	if (_verbose > 1) Serial.println("Creating dataset...");
	_nData = nData;
	_xmin.clear();
	_ymin.clear();
	_xmax.clear();
	_ymax.clear();

	for (size_t j = 0; j < _nInputs; ++j) {
		_xmin.push_back(MAX_float);
		_xmax.push_back(MIN_float);
		for (size_t i = 0; i < _nData; ++i) {
			float xx = x0(i,j);
			if (xx < _xmin[j]) _xmin[j] = xx;
			if (xx > _xmax[j]) _xmax[j] = xx;
		}
	}
	for (size_t j = 0; j < _nClasses; ++j) {
		_ymin.push_back(MAX_float);
		_ymax.push_back(MIN_float);
		for (size_t i = 0; i < _nData; ++i) {
			if (_enSoftmax && _labelSmoothing) {
				// Label smoothing is a way of adding noise at the output targets, aka labels.
				float epsilon = 0.001f;
				if (y0(i,j) == 0.0f) y0(i,j) = epsilon / (_nClasses - 1.0f);
				else                 y0(i,j) = 1.0f - epsilon;
			}
			float yy = y0(i,j);
			if (yy < _ymin[j]) _ymin[j] = yy;
			if (yy > _ymax[j]) _ymax[j] = yy;
		}
	}

	if (_verbose > 0) Serial.println("Processing dataset");
	if (_verbose > 1) {
		Serial.printf("\tDataset contains %d data\n", _nData);
		Serial.print("\tMin value of x: ");
		for (size_t j = 0; j < _nInputs; ++j)  Serial.printf("%9.3f ", _xmin[j]); Serial.println();
		Serial.print("\tMax value of x: ");
		for (size_t j = 0; j < _nInputs; ++j)  Serial.printf("%9.3f ", _xmax[j]); Serial.println();
		Serial.printf("\tMin value of y: ");
		for (size_t j = 0; j < _nClasses; ++j) Serial.printf("%9.3f ", _ymin[j]); Serial.println();
		Serial.printf("\tMax value of y: ");
		for (size_t j = 0; j < _nClasses; ++j) Serial.printf("%9.3f ", _ymax[j]); Serial.println();
	}
}

// Second case: dataset is an array
// Implies 1 input neuron and 1 output neuron
void MLP::createDatasetFromArray (MLMatrix<float> &x0, MLMatrix<float> &y0, const float *x, const float *y, const int nData)
{
	_nData = nData;
	_xmin.clear();
	_ymin.clear();
	_xmax.clear();
	_ymax.clear();

	_xmin.push_back(MAX_float);
	_xmax.push_back(MIN_float);
	_ymin.push_back(MAX_float);
	_ymax.push_back(MIN_float);

	std::vector<float> a, b;
	for (size_t i = 0; i < _nData; ++i) {
		if (x[i] < _xmin[0]) _xmin[0] = x[i];
		if (x[i] > _xmax[0]) _xmax[0] = x[i];
		if (y[i] < _ymin[0]) _ymin[0] = y[i];
		if (y[i] > _ymax[0]) _ymax[0] = y[i];
		a.push_back(x[i]);
		b.push_back(y[i]);
	}
	x0 = a;
	y0 = b;

	if (_verbose > 0) Serial.printf("Creating dataset from arrays of size %d\n", _nData);
	if (_verbose > 1) {
		Serial.printf("\tMin value of x: %f\n", _xmin[0]);
		Serial.printf("\tMax value of x: %f\n", _xmax[0]);
		Serial.printf("\tMin value of y: %f\n", _ymin[0]);
		Serial.printf("\tMax value of y: %f\n", _ymax[0]);
	}
}

// Third case: dataset is a vector
// Implies 1 input and 1 output
void MLP::createDatasetFromVector (MLMatrix<float> &x0, MLMatrix<float> &y0, const std::vector<float>x, const std::vector<float>y)
{
	_nData = x.size();
	_xmin.clear();
	_ymin.clear();
	_xmax.clear();
	_ymax.clear();

	_xmin.push_back(MAX_float);
	_xmax.push_back(MIN_float);
	_ymin.push_back(MAX_float);
	_ymax.push_back(MIN_float);

	for (size_t i = 0; i < _nData; ++i) {
		if (x[i] < _xmin[0]) _xmin[0] = x[i];
		if (x[i] > _xmax[0]) _xmax[0] = x[i];
		if (y[i] < _ymin[0]) _ymin[0] = y[i];
		if (y[i] > _ymax[0]) _ymax[0] = y[i];
	}
	x0 = x;
	y0 = y;

	if (_verbose > 0) Serial.printf("Creating dataset from vectors of size %d\n", _nData);
	if (_verbose > 1) {
		Serial.printf("\tMin value of x: %f\n", _xmin[0]);
		Serial.printf("\tMax value of x: %f\n", _xmax[0]);
		Serial.printf("\tMin value of y: %f\n", _ymin[0]);
		Serial.printf("\tMax value of y: %f\n", _ymax[0]);
	}
}

void MLP::shuffleDataset (MLMatrix<float> &x, MLMatrix<float> &y, uint16_t from, uint16_t to)
{
	if (_verbose > 1) Serial.println("Shuffling dataset...");
	if (from >= to || from > _nData || to > _nData) {
		Serial.printf ("Error: cannot shuffle from %d to %d", from, to);
		while (1);
	}
	// if (to >= _nData) to = _nData;
	// if (from >= _nData) from = to - 1;

	for (size_t i = 0; i < 5 * _nData; ++i) {
		int k = random(from, to);
		int m = random(from, to);
		MLMatrix<float> Xk(1, _nInputs, 0);
		MLMatrix<float> Yk(1, _nClasses, 0);
		Xk = x.row(k);
		Yk = y.row(k);
		for (size_t j = 0; j < _nInputs; ++j)  {
			x(k,j) = x(m,j);
			x(m,j) = Xk(0,j);
		}
		for (size_t j = 0; j < _nClasses; ++j) {
			y(k,j) = y(m,j);
			y(m,j) = Yk(0,j);
		}
	}
}

/* Normalize the dataset
		Input data is always normalized, output data is normalized if
		the output activation is not Softmax

		norm is a flag to choose the normalization:
		0: no normalization
		1: data is set to interval [0 - 1]
		2: data is set to interval [-1 - 1]
		3: (data - mean) / std_dev
*/
void MLP::normalizeDataset (MLMatrix<float> &x, MLMatrix<float> &y, const uint8_t norm)
{
	Serial.printf("Normalizing the dataset (option %d)\n", norm);
	_norm = norm;
	switch (_norm) {
		case 0: break;
		case 1: {
			for (size_t i = 0; i < _nData; ++i) {
				for (size_t j = 0; j < _nInputs; ++j) {
					x(i,j) = (x(i,j) - _xmin[j]) / (_xmax[j] - _xmin[j]);
					// if (i==0) Serial.printf("input %d : min %f max %f\n",j,_xmin[j], _xmax[j]);
				}
				if (_activations[_nLayers - 2] != SOFTMAX) {
					for (size_t j = 0; j < _nClasses; ++j) {
						y(i,j) = (y(i,j) - _ymin[j]) / (_ymax[j] - _ymin[j]);
					// if (i==0) Serial.printf("class %d : min %f max %f\n",j,_ymin[j], _ymax[j]);
					}
				}				
			}
			break;	
		}
		case 2: {
			for (size_t i = 0; i < _nData; ++i) {
				for (size_t j = 0; j < _nInputs; ++j)
					x(i,j) = ((x(i,j) - _xmin[j]) / (_xmax[j] - _xmin[j]) - 0.5f) * 2.0f;
				if (_activations[_nLayers - 2] != SOFTMAX) {
					for (size_t j = 0; j < _nClasses; ++j)
						y(i,j) = ((y(i,j) - _ymin[j]) / (_ymax[j] - _ymin[j]) - 0.5f) * 2.0f;
				}
			}
			break;		
		}
		case 3: {
			_xMean.clear();
			_xStdev.clear();
			_yMean.clear();
			_yStdev.clear();

			for (size_t j = 0; j < _nInputs; ++j) {
				MLMatrix<float> vx(_nData,1,0);
				for (size_t i = 0; i < _nData; ++i) vx(i,0) = x(i,j);
				_xMean.push_back(vx.mean());
				_xStdev.push_back(vx.stdev(_xMean[j]));
				for (size_t i = 0; i < _nData; ++i)
					x(i,j) = (x(i,j) - _xMean[j]) / _xStdev[j];	
			}
			if (_activations[_nLayers - 2] != SOFTMAX) {
				for (size_t j = 0; j < _nClasses; ++j) {
					MLMatrix<float> vy(_nData,1,0);
					for (size_t i = 0; i < _nData; ++i) vy(i,0) = y(i,j);
					_yMean.push_back(vy.mean());
					_yStdev.push_back(vy.stdev(_yMean[j]));
					for (size_t i = 0; i < _nData; ++i)
							y(i,j) = (y(i,j) - _yMean[j]) / _yStdev[j];	
				}
			}
			break;
		}
		default:
			Serial.printf("Normalization error : unknown argument (%u)\n", norm);
			break;
	}

}

/* Normalize a matrix
		norm is a flag to choose the normalization:
		0: no normalization
		1: data is set to interval [0 - 1]
		2: data is set to interval [-1 - 1]
		3: (data - mean) / std_dev
*/
void MLP::normalize(MLMatrix<float> &x, const uint8_t norm)
{
	unsigned rows = x.get_rows();
	unsigned cols = x.get_cols();

	if (cols == 1) { // data is a one-column vector
		switch (norm) {
			case 0: break;
			case 1: {
				for (size_t i = 0; i < rows; i++) 
					x(i,0) = (x(i,0) - _xmin[i]) / (_xmax[i] - _xmin[i]);
				break;	
			}
			case 2: {
				for (size_t i = 0; i < rows; i++)
					x(i,0) = ((x(i,0) - _xmin[i]) / (_xmax[i] - _xmin[i]) - 0.5f) * 2.0f;
				break;		
			}
			case 3: {
				for (size_t i = 0; i < rows; i++) 
					x(i,0) = (x(i,0) - _xMean[i]) / _xStdev[i];
				break;
			}
			default:
				Serial.printf("Normalization error : invalid argument (%u)\n", norm);
				break;
		}

	} else { // data is a one-line vector

		switch (norm) {
			case 0: break;
			case 1: {
				for (size_t i = 0; i < cols; ++i) 
					x(0,i) = (x(0,i) - _xmin[i]) / (_xmax[i] - _xmin[i]);
				break;	
			}
			case 2: {
				for (size_t i = 0; i < cols; i++)
					x(0,i) = ((x(0,i) - _xmin[i]) / (_xmax[i] - _xmin[i]) - 0.5f) * 2.0f;
				break;		
			}
			case 3: {
				for (size_t i = 0; i < cols; ++i) 
					x(0,i) = (x(0,i) - _xMean[i]) / _xStdev[i];
				break;
			}
			default:
				Serial.printf("Normalization error : invalid argument (%u)\n", norm);
				break;
		}

	}
}

// void MLP::deNorm (std::vector<std::vector<float> > &y, const uint8_t norm)
// {
// 	unsigned rows = y.size();
// 	unsigned cols = y[0].size();
// 	if (_verbose > 1) Serial.printf("deNorm: cols %d rows %d\n",cols,rows);
// 	switch (norm) {
// 		case 0: break;
// 		case 1: {
// 			for (size_t i = 0; i < rows; ++i)
// 				for (size_t j = 0; j < cols; ++j)
// 					y(i,j) = (y(i,j) * (_ymax[j] - _ymin[j])) + _ymin[j];
// 			break;
// 		}
// 		case 2: {
// 			for (size_t i = 0; i < rows; i++)
// 				for (size_t j = 0; j < cols; ++j)
// 					y(i,j) = ((y(i,j) / 2.0f) + 0.5f) * (_ymax[j] - _ymin[j]) + _ymin[j];
// 			break;		
// 		}
// 		case 3: {
// 			for (size_t i = 0; i < rows; i++) 
// 				for (size_t j = 0; j < cols; ++j)
// 					y(i,j)= (y(i,j) * _yStdev[j]) + _yMean[j];
// 			break;
// 		}
// 		default:
// 			Serial.printf("Normalization error : invalid argument (%u)\n", norm);
// 			break;
// 	}
// }

void MLP::deNorm(MLMatrix<float> &y, const uint8_t norm)
{
	unsigned rows = y.get_rows();
	unsigned cols = y.get_cols();

	if (cols == 1) { // data is a one-column vector
		switch (norm) {
			case 0: break;
			case 1: {
				for (size_t i = 0; i < rows; i++) 
					y(i,0) = (y(i,0) * (_ymax[i] - _ymin[i])) + _ymin[i];
				break;	
			}
			case 2: {
				for (size_t i = 0; i < rows; i++) {
					y(i,0) = ((y(i,0) / 2.0f) + 0.5f) * (_ymax[i] - _ymin[i]) + _ymin[i];
				}
				break;		
			}
			case 3: {
				for (size_t i = 0; i < rows; i++) 
					y(i,0) = (y(i,0) * _yStdev[i]) + _yMean[i];
				break;
			}
			default:
				Serial.printf("Normalization error : invalid argument (%u)\n", norm);
				break;
		}

	} else { // data is a one-line vector

		switch (norm) {
			case 0: break;
			case 1: {
				for (size_t i = 0; i < cols; ++i) 
					y(0,i) = (y(0,i) * (_ymax[i] - _ymin[i])) + _ymin[i];
				break;	
			}
			case 2: {
				for (size_t i = 0; i < cols; i++)
					y(0,i) = ((y(0,i) / 2.0f) + 0.5f) * (_ymax[i] - _ymin[i]) + _ymin[i];
				break;		
			}
			case 3: {
				for (size_t i = 0; i < cols; ++i) 
					y(0,i) = (y(0,i) * _yStdev[i]) + _yMean[i];
				break;
			}
			default:
				Serial.printf("De-normalization error : invalid argument (%u)\n", norm);
				break;
		}
	}
}

/*
    readCsvFromSpiffs (filename, x, y)
    Reads the dataset from a csv file on LITTLEFS
    nData : number of lines of the file
    A line is made of: x1, x2, x3 ... xN, Out
    where N is the number of neurons of the input layer
*/
int MLP::readCsvFromSpiffs (const char* const path, MLMatrix<float>& x0, MLMatrix<float>& y0)
{
	if (_verbose > 1) Serial.printf ("Opening file %s\n", path);
	File file = LITTLEFS.open(path);
	if (!file || file.isDirectory()) {
		Serial.printf("%s - failed to open file for reading\n", path);
		return 0;
	}
	char buffer[500];
	char * pch;

	// First line : number of cols = 1 + number of input neurons
	int nData = 1;
	int i = 0;
	while (file.available()) { // Read first line
		char c = file.read();
		buffer[i++] = c;
		if (c == 10) break; // CR
		if (i > 500) {
			Serial.println("Error reading file : line is too long (500 characters max)!");
			while (1);
		}
	}
	buffer[i] = NULL;
	if (_verbose > 2) Serial.println(buffer);

	std::vector<float> X, Y;
	int nCols = 0;
	pch = strtok (buffer, ",;");
	while (pch != NULL) { // Count columns
		float data = atof(pch);
		if (nCols < _neurons[0]) X.push_back(data);
		else Y.push_back(data);
		++nCols;
		pch = strtok (NULL, ",;");
	}
	if (nCols != _neurons[0] + 1) {
		Serial.printf("Problem reading file line %d\n", nData);
		Serial.printf("Read %d columns, and %d input neurons require %d columns\n",
			nCols, _neurons[0], _neurons[0] + 1);
		while (1);
	}
	if (_verbose > 1) Serial.printf ("Read first line: found %d columns\n", nCols);
	std::vector<std::vector<float> > Vx, Vy;
	Vx.push_back(X);
	Vy.push_back(Y);

	// Next lines
	while (file.available()) {
		i = 0;
		char c = file.read();
		while (c != 10) { // CR
			buffer[i++] = c;
			c = file.read();
		}
		X.clear();
		Y.clear();
		buffer[i] = NULL;
		pch = strtok (buffer, ",;");
		for (size_t i = 0; i < nCols; i++) {
			float data = atof(pch);
			if (i < nCols - 1) X.push_back(data);
			else Y.push_back(data);
			pch = strtok (NULL, ",;");
		}
		Vx.push_back(X);
		Vy.push_back(Y);
		++nData;
	}

	if (_verbose > 1) Serial.printf ("Read file complete: found %d lines\n", nData);
	x0.setSize(nData, _nInputs);
	y0.setSize(nData, _nClasses);
	x0 = Vx;
	if (_enSoftmax) 
		for (size_t i = 0; i < Vy.size(); ++i) y0(i,int(Vy[i][0])) = 1; // One hot encoder
	else y0 = Vy;

	createDataset (x0, y0, nData);
	if (_verbose > 0) Serial.printf("Read %d data of %d input\n", nData, _neurons[0]);
	return _nData;
	}

/************************************

		Network functions	 

*************************************/
MLMatrix<float> MLP::forward (MLMatrix<float> x, bool onlyInference)
{
	if (_verbose > 2) Serial.println("Forward...");
	// 1: Clear workspace
	_a.clear();
	_dropoutMasks.clear();
	MLMatrix<float> yhat;
	if (!onlyInference) _a.push_back(x); // saved for backward pass

	// 2: Forward pass
	for (size_t k = 0; k < _nLayers - 1; ++k) {
		if (_dropout) { // Apply dropout
			MLMatrix<uint8_t> Mask = x.dropout(_dropout_prob);
			_dropoutMasks.push_back(Mask);
			x /= _dropout_prob;
		}

		yhat = Weights[k] * x + Biases[k];
		if (_quadLayers) {
			MLMatrix<float> x2 = x.square();
			yhat = yhat + Weights2[k] * x2;
		} //else // No activation for quadratic layers
		yhat = activation(yhat, _activations[k]);
		if (!onlyInference) _a.push_back(yhat); // saved for backward pass
		x = yhat;
	}
	if (_verbose > 2) yhat.print();
	return yhat;
}

MLMatrix<float> MLP::activation(MLMatrix<float> x, const uint8_t activNumber)
{
	uint8_t rows = x.get_rows();
	MLMatrix<float> result(rows, 1, 0.0f);
	switch (activNumber) {
		case RELU:
			for (size_t i = 0; i< rows; ++i) result(i,0) = ReLu(x(i,0));
			break;
		case SIGMOID:
			for (size_t i = 0; i< rows; ++i) result(i,0) = Sigmoid(x(i,0));
			break;
		case SIGMOID2:
			for (size_t i = 0; i< rows; ++i) result(i,0) = Sigmoid2(x(i,0));
			break;
		case TANH:
			for (size_t i = 0; i< rows; ++i) result(i,0) = Tanh(x(i,0));
			break;
		case ID:
			for (size_t i = 0; i< rows; ++i) result(i,0) = Id(x(i,0));
			break;
		case SOFTMAX:
			result = SoftMax(x);
			break;
		case LEAKYRELU:
			for (size_t i = 0; i< rows; ++i) result(i,0) = LeakyReLu(x(i,0));
			break;
		case ELU:
			for (size_t i = 0; i< rows; ++i) result(i,0) = ELu(x(i,0));
			break;
		case SELU:
			for (size_t i = 0; i< rows; ++i) result(i,0) = SeLu(x(i,0));
			break;
		case RELU6:
			for (size_t i = 0; i< rows; ++i) result(i,0) = ReLu6(x(i,0));
			break;
		case SWISH:
			for (size_t i = 0; i< rows; ++i) result(i,0) = Swish(x(i,0));
			break;
		default:
			Serial.printf("Unknown activation (%d)\n", activNumber);
	}
	return result;
}

MLMatrix<float> MLP::dActivation(MLMatrix<float> x, const uint8_t activNumber)
{
	uint8_t rows = x.get_rows();
	MLMatrix<float> result(rows, 1, 1.0f);
	switch (activNumber) {
		case RELU:
			for (size_t i = 0; i< rows; ++i) result(i,0) = dReLu(x(i,0));
			break;
		case SIGMOID:
			for (size_t i = 0; i< rows; ++i) result(i,0) = dSigmoid(x(i,0));
			break;
		case SIGMOID2:
			for (size_t i = 0; i< rows; ++i) result(i,0) = dSigmoid2(x(i,0));
			break;
		case TANH:
			for (size_t i = 0; i< rows; ++i) result(i,0) = dTanh(x(i,0));
			break;
		case ID:
			for (size_t i = 0; i< rows; ++i) result(i,0) = dId(x(i,0));
			break;
		case SOFTMAX:
		// Return vector full of 1
			break;
		case LEAKYRELU:
			for (size_t i = 0; i< rows; ++i) result(i,0) = dLeakyReLu(x(i,0));
			break;
		case ELU:
			for (size_t i = 0; i< rows; ++i) result(i,0) = dELu(x(i,0));
			break;
		case SELU:
			for (size_t i = 0; i< rows; ++i) result(i,0) = dSeLu(x(i,0));
			break;
		case RELU6:
			for (size_t i = 0; i< rows; ++i) result(i,0) = dReLu6(x(i,0));
			break;
		case SWISH:
			for (size_t i = 0; i< rows; ++i) result(i,0) = dSwish(x(i,0));
			break;
		default:
			Serial.printf("Unknown (d)activation (%d)\n", activNumber);
	}
	return result;
}

float MLP::error (MLMatrix<float> y, MLMatrix<float> yhat) const
{
	if (_verbose > 2) Serial.print("Error... ");
	float err = 0.0f;
	int idh0, idh1, idy0, idy1;
	if (_enSoftmax) {
		yhat.indexMax(idh0, idh1);
		y.indexMax(idy0, idy1);
		err = abs(idh0 - idy0);
	} else {
		for (size_t c = 0; c < _nClasses; ++c) 
			err += abs (y(c, 0) - yhat(c, 0));
	}
	return err;
}
// ************************************************************************
//
// Set the hyperparameters according to the epoch number and other stuff...
//
// ************************************************************************
void MLP::heuristics (int epoch, int maxEpochs, bool _better)
{
	static bool Aup = true;
	static bool Gup = false;
	static uint8_t nbRestore = 0;

// Restore or change weights if too many epochs without improvement
	if (epoch - _lastBestEpoch > maxEpochs / 10) {
		if (_changeWeights) {
			Serial.println("Setting new random weights");
			initWeights(); // New random weights
		} else {
			Serial.println("Restoring last saved weights");
			restoreWeights(); // Restore weights
			++nbRestore;
			// Apply small random changes to weights (25% chance)
			if ((rand01() < 0.25f && _mutateWeights) || (nbRestore == 3)) {
				nbRestore = nbRestore % 3; // force weights changes after 3 weight restorations
				float amplitude = 0.1f; // Change values up to 10%
				Serial.printf("Random change to weights (amplitude %.1f%%)\n", amplitude * 100);
				for (size_t k = 0; k < _nLayers - 1; ++k)
					Weights[k].randomChange(amplitude);
			}
		}
		_lastBestEpoch = epoch;
		if (_bestEta) {
			Serial.println("Switching best eta to false");
			_bestEta = false;
		}
		// and set LR to original value
		// Serial.println("Setting LR to initial value");
		// _eta = _eta0;
	}

// Apply learning rate variation (lin / log)
	if (_changeLRlin) {
		_eta = _eta0 + epoch * (_etaMin - _eta0) / maxEpochs;
		if (_verbose > 1) Serial.printf ("Heuristics: LR = %f, mom = %f\n", _eta, _momentum);
	}

	if (_changeLRlog) {
		float logLR = _logLRmax + epoch * (_logLRmin - _logLRmax) / maxEpochs;
		_eta = pow(10.0f, logLR);
		if (_verbose > 1) Serial.printf ("Heuristics: LR = %f, mom = %f\n", _eta, _momentum);
	}

// Apply quadratic momentum variation (from 0.5 to 0.99)
	if (_varMom) {
		// _momentum = _minMom + epoch * (_maxMom - _minMom) / maxEpochs;
		_momentum = _minMom + epoch * epoch * (_maxMom - _minMom) / maxEpochs / maxEpochs;
		if (_verbose > 1) Serial.printf ("Heuristics: LR = %f, mom = %f\n", _eta, _momentum);
	}

// Apply small random changes to weights (3% chance)
	if (rand01() < 0.03f && _mutateWeights) {
		float amplitude = 0.025f;
		if (_verbose > 0) Serial.printf("Random change to weights (amplitude %.1f%%)\n", amplitude * 100);
		for (size_t k = 0; k < _nLayers - 1; ++k)
			Weights[k].randomChange(amplitude);
	}

// Apply other changes
	// (nDots is the number of epochs since last best epoch)
	if (_currError / _stopError > 3 && _nDots > 5) {
		_nDots = 0;

	// Add neurons
	if (_prune_topk || _prune_neurons || _prune_train) {
		uint8_t nbNeurons = random(1, 6);
		for (unsigned i = 0; i < nbNeurons; ++i) {
			int layer = random(1, _nLayers - 1);
			Serial.printf("Heuristics: adding a neuron to layer %d\n", layer);
			addNeuron(layer);
		}
		size();
	 _maxEpochs += 5;
	 Serial.printf("Adding 5 epochs : total epoch number now is %d\n", _maxEpochs);
	}

	// Momentum variation
		if (rand01() < 0.25f && _changeMom && !_varMom) {
			float alpha = _momentum;
			if (alpha <= _minAlpha || alpha >= _maxAlpha) Aup = !Aup;
			if (Aup) _momentum = alpha + 0.15;
			else _momentum = alpha - 0.15;
			if (_verbose > 0) Serial.printf ("Heuristics: changing momentum to %f\n", _momentum);
		}
	
	// Sigmoid gain variation
		if (rand01() < 0.25f && _changeGain) {
			float gain = _gain;
			if (gain <= _minGain || gain >= _maxGain) Gup = !Gup;
			if (Gup) _gain = gain + 0.15;
			else _gain = gain - 0.15;
			if (_verbose > 0) Serial.printf ("Heuristics: changing sigmoid gain to %.2f\n", _gain);
		}
	
		// if (_bestEta) {
		// 	Serial.println("Switching best eta to false");
		// 	_bestEta = false;
		// }

// Display values
		if (_verbose > 1) {
			Serial.printf ("Heuristics: LR = %f, mom = %f sigmoid gain= %f\n", _eta, _momentum, _gain);
		}
	}
}

// Increase the score of the neurons in top K of each layer
void MLP::topKCount (float threshold, int layer, MLMatrix<float>& b)
{
	int rows = b.get_rows();
	int cols = b.get_cols();
	// Serial.printf("topK count: layer %d dimension %d %d, threshold %f\n", layer, rows, cols, threshold);
	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			if (b(i,j) > threshold) b_topK[layer][i] += 1;
		}
	}
}

// Back propagation
void MLP::backward (const MLMatrix<float> yhat, const MLMatrix<float> y, const int d)
{
	if (_verbose > 2) Serial.println("Backward...");
	MLMatrix<float> delta;
	MLMatrix<float> Weights_old;
	MLMatrix<float> Weights_old2, c2, delta2;

// http://neuralnetworksanddeeplearning.com/chap2.html#the_backpropagation_algorithm
	for (size_t k = _nLayers - 1; k > 0; --k) {
		if (k == _nLayers - 1) {
			delta = yhat - y; // last layer (MSE or Cross Entropy)
			if (_quadLayers) delta2 = delta;
		} else {
			delta = Weights_old.transpose() * delta;
			if (_quadLayers) delta2 = Weights_old2.transpose() * delta2;
		}
		if (_verbose > 2) delta.print();

		if (_dropout && k != _nLayers - 1) { // Consider dropout
			MLMatrix<uint8_t> Mask = _dropoutMasks.back();
			_dropoutMasks.pop_back();
			if ( delta.get_rows() != Mask.get_rows() || delta.get_cols() != Mask.get_cols() ) { // matrices of different sizes
		    Serial.printf("Hadamard product error: dimensions do not match (%d, %d).(%d, %d)", delta.get_rows(),delta.get_cols(),Mask.get_rows(),Mask.get_cols());
		    while(1);
		  }
		  MLMatrix<float> result(delta);
		  for ( unsigned i = 0; i < delta.get_rows(); ++i )
      	for ( unsigned j = 0; j < delta.get_cols(); ++j )
        	result(i,j) = delta(i,j) * Mask(i,j);
			delta = result / _dropout_prob;
		}

		MLMatrix<float> b = dActivation(_a[k], _activations[k - 1]);
		delta = delta.Hadamard(b);

		Weights_old = Weights[k - 1];
		MLMatrix<float> c = delta * _a[k - 1].transpose();
		if (_verbose > 2) delta.print();
		if (_verbose > 2) c.print();

		if (_quadLayers) {
			delta2 = delta2.Hadamard(b);
			Weights_old2 = Weights2[k - 1];
			c2 = delta2 * _a[k - 1].transpose();
		}

		// Manage topK pruning
		if (_prune_topk) {
			// inspired from https://arxiv.org/ftp/arxiv/papers/1711/1711.06528.pdf
			std::vector<float> V = c.sortValues();
			int nbW = int(V.size() * _topKpcent);
			float threshold = V[nbW];
			// Clip to zero gradient values under threshold
			int nbClipW = c.clipToZero(threshold);
			int nbClipB = delta.clipToZero(threshold);
			// Serial.printf("Gradient clipping:  nbW %d thresh %f clipped weights %d, biases %d\n",nbW, threshold, nbClipW, nbClipB);
			std::vector<float> Vb = b.sortValues();
			int nbWb = int(Vb.size() * 0.75);
			threshold = Vb[nbWb];
			if (k != _nLayers - 1 && k != 0) topKCount(threshold, k, b);
		}

		// Compute gradients
		if (d == 0) {
			dWeights.push_back(c);
			if (_quadLayers) dWeights2.push_back(c2);
			dBiases.push_back(delta);
		}	else {
			dWeights[_nLayers - 1 - k] += c;
			if (_quadLayers) dWeights2[_nLayers - 1 - k] += c2;
			dBiases[_nLayers - 1 - k] += delta;
		}
		if (_verbose > 2) dBiases[_nLayers - 1 - k].print();
  }
}

// Update weights & biases
void MLP::update (const int batchSize)
{
	float eta = _eta;
	for (size_t k = _nLayers - 1; k > 0; --k) {
		// Testing layer specific learning rates (aka differential LR): not convincing
		// eta = _eta * (1.0f + (_nLayers - k - 1.0f) / (_nLayers - 2.0f)); // LR decreasing with deeper layers
		// eta = _eta * (1.0f + (k - 1.0f) / (_nLayers - 2.0f)); // LR increasing with deeper layers (better)
		MLMatrix<float> W = Weights[k - 1];
		// MLMatrix<float> B = Biases[k - 1];

// Gradient clipping
		if (_gradClip && !_firstRun) {
			dBiases[_nLayers - 1 - k].clipMax(_gradClipValue);
			dWeights[_nLayers - 1 - k].clipMax(_gradClipValue);
			if (_quadLayers) dWeights2[_nLayers - 1 - k].clipMax(_gradClipValue);
		}

// Rescale gradient
		if (_gradScaling && !_firstRun) {
			bool zeroNorm = dBiases[_nLayers - 1 - k].normScale2(_gradScale);
			if (zeroNorm) {
				Serial.printf("Warning: biases gradients are zero (layer %d)\n", k);
				// dBiases[_nLayers - 1 - k].print();
			}
			zeroNorm = dWeights[_nLayers - 1 - k].normScale2(_gradScale);
			if (_quadLayers) zeroNorm += dWeights2[_nLayers - 1 - k].normScale2(_gradScale);
			if (zeroNorm)	{
				Serial.printf("Warning: weights gradients are zero (layer %d)\n", k);
				// dWeights[_nLayers - 1 - k].print();
			}
		}

// Update weights
		Biases[k - 1]  -= dBiases[_nLayers - 1 - k]  * (eta / batchSize);
		Weights[k - 1] -= dWeights[_nLayers - 1 - k] * (eta / batchSize);
		if (_quadLayers) Weights2[k - 1] -= dWeights2[_nLayers - 1 - k] * (eta / batchSize);
		if (!_firstRun) {
			Biases[k - 1]  -= dBiasesOld[_nLayers - 1 - k]  * (_momentum / batchSize);
			Weights[k - 1] -= dWeightsOld[_nLayers - 1 - k] * (_momentum / batchSize);
			if (_quadLayers) Weights2[k - 1] -= dWeightsOld2[_nLayers - 1 - k] * (_momentum / batchSize);
		}

// Apply regularization penalty
		if (_regulL1)  {
			// Biases[k - 1]  -= B.sgn() * _lambdaRegulL1 * (eta / batchSize);
			Weights[k - 1] -= W.sgn() * _lambdaRegulL1 * (eta / batchSize);
		}
		if (_regulL2)  {
			// Biases[k - 1]  -= B * _lambdaRegulL2 * (eta / batchSize);
			Weights[k - 1] -= W * _lambdaRegulL2 * (eta / batchSize);
		}

// Force weights to zero if lower than threshold
		if (_zeroWeights) {
			int nbBiasClip    = Biases[k - 1].clipToZero(_zeroThreshold);
			int nbWeightClip  = Weights[k - 1].clipToZero(_zeroThreshold);
			int nbWeight2Clip = (_quadLayers) ? Weights2[k - 1].clipToZero(_zeroThreshold) : 0;
			if (nbBiasClip + nbWeightClip + nbWeight2Clip == 0 && _verbose > 1) 
				Serial.printf ("Warning: clipping threshold (%.4f) too low, no weights to clip...\n", _zeroThreshold);
		}
	}
	_firstRun = false;
}

void MLP::searchEta (MLMatrix<float> x, MLMatrix<float> y, float Err)
{
	MLMatrix<float> x0 = x;
	_eta = _eta0;
	float prevErr = Err;
	float err;
	while (_eta > _etaMin) {
		_firstRun = true;
		update (_batchSize);
		MLMatrix<float> yhat = forward (x, true);
		x = x0;
		err = error (y, yhat);
		// Serial.printf("prev %f -- new err %f \n",prevErr, err);
		while (err < prevErr && err > _minError * 0.75f) {
			prevErr = err;
			_firstRun = true;
			update (_batchSize);
			yhat = forward (x, true);
			err = error (y, yhat);
			x = x0;
			// Serial.printf("searchEta : eta %f error %f (<? %f)\n",_eta, err, prevErr);
		}
		_eta /= 2;
	}
	if (_verbose > 1) Serial.printf("searchEta: LR = %f, error is %f\n", _eta, err);
}

// Test 30 random sets of weights and use the best one for optimization
void MLP::searchBestWeights(const MLMatrix<float> x0, const MLMatrix<float> y0)
{
	Serial.println("Searching best starting weights");
	byte N = 30; // Number of random tests
	float minErr = 100000.0f;

	MLMatrix<float> x(_nInputs,  1, 0.0f);  // Input array
	MLMatrix<float> y(_nClasses, 1, 0.0f);  // Ground truth
	
	for (size_t i = 0; i < N; ++i) {
		float err = 0.0f;
		// Change the initialization range
		float vmax = 0.7f - 0.6f * i / N;
		float vmin = - vmax;
		for (size_t d = 0; d < _nTrain / 3; ++d) {
			for (size_t i = 0; i < _nInputs; ++i)  x(i, 0) = x0(d,i);
			for (size_t c = 0; c < _nClasses; ++c) y(c, 0) = y0(d,i);
			initWeights(vmin, vmax);
// Forward pass
			MLMatrix<float> yhat = forward (x, true);
// Compute error
			err += error (y, yhat);
		}
		err /= (_nTrain / 3);
		if (_verbose > 1) Serial.printf ("Trial number %d, error %f (%f)\n",i,err,minErr);

		if (err < minErr) {
			minErr = err;
			Serial.printf("--> Found better weights (error = %.4f, init = %.3f)\n",minErr, vmax);
			saveWeights();
		}
	}
	// restoreWeights();
}

// ************************************************************************
//
//     Optimize the network
//
// ************************************************************************
void MLP::run(MLMatrix<float> x0, MLMatrix<float> y0, int maxEpochs, int batchSize, float stopError)
{
	if (!_changeBSize) Serial.printf("Batch size = %d\n",batchSize);
	Serial.printf("Stopping if error < %.3f\n",stopError);
	Serial.println("\nRunning optimization...");

// Prepare dataset
	if (!_datasetSplit) {
		_nTrain = _nData * rTrain;
		_nValid = _nData * rValid;
		_nTest  = _nData * rTest;
		Serial.printf ("Dataset split in: %d train + %d validation + %d test data\n", _nTrain, _nValid, _nTest);
		_datasetSplit = true;
	}
	// Shuffle dataset
	shuffleDataset (x0, y0, 0, _nData);

	// Pre-train on a subset of the dataset (20%) for fastest beginning
	uint16_t savenTrain, savenValid, nimprove = 0;
	bool subTrain;
	if (_dataSubset) {
		savenTrain = _nTrain;
		savenValid = _nValid;
		_nTrain /= 5;
		_nValid /= 5;
		subTrain = true;
	}

// Prepare topK pruning
	if (_prune_topk) {
		b_topK.resize(_nLayers - 1);
		for (size_t i = 0; i < _nLayers - 1; ++i) b_topK[i].resize(_neurons[i], 0.0f);
	}

// Initialize weights
	if (_initialize) {
		if (_verbose > 0) Serial.println("Creating a new network");
		(_selectWeights) ? searchBestWeights(x0, y0) : initWeights();
	}

// Estimate training duration
	uint32_t duration = estimateDuration (maxEpochs);
	Serial.printf("Estimated maximum duration : %.2f s for %d epochs\n", duration/1000.0f, maxEpochs);

// Variables
	MLMatrix<float> x(_nInputs,  1, 0.0f);  // Input array
	MLMatrix<float> y(_nClasses, 1, 0.0f);  // Ground truth
	bool _better = false;
	_batchSize = batchSize;
	if (_batchSize >= _nTrain) _batchSize = max(1, _nTrain / 5);
	_stopError = stopError;
	_firstRun = true;
	_maxEpochs = maxEpochs;
	byte pruneEpochs = 0;

	// Epochs loop...
	int epoch = 1;
	unsigned long chrono = millis();
	while (epoch <= _maxEpochs) {
		if (_verbose > 1) Serial.printf("Epoch %d\n", epoch);

// Check if pre-train phase is finished
		if (_dataSubset && subTrain) {
			if (epoch > _maxEpochs / 3 || nimprove > 4) { // Back to entire dataset
				_nTrain = savenTrain;
				_nValid = savenValid;
				subTrain = false;
				_minError *= 10.0f;
				Serial.println("Now training on the entire dataset");
			}
		}

// Manage hyper-parameters
		if (epoch > 1) heuristics(epoch, _maxEpochs, _better);

		_prevError = _currError;
		float totalError = 0.0f;
		int data = 0;

		if (_changeBSize) { // (heuristics) change batch size
			float coef = _maxBS + epoch * (_minBS - _maxBS) / _maxEpochs;
			batchSize = int(coef * _nTrain);
			if (batchSize < 1) batchSize = 1;
			if (_verbose > 0) Serial.printf ("Batch size = %d\n", batchSize);
		}
// Loop over training set
		while (data < _nTrain) {
			dWeightsOld = dWeights;
			if (_quadLayers) dWeightsOld2 = dWeights2;
			dBiasesOld = dBiases;
			dWeights.clear();
			if (_quadLayers) dWeights2.clear();
			dBiases.clear();
			if (_verbose > 2) Serial.printf ("Data number %d (%d)\n", data, batchSize);
			float err;

// Loop over minibatch
			for (size_t d = 0; d < batchSize; ++d) {
				for (size_t i = 0; i < _nInputs; ++i)  x(i, 0) = x0(data + d,i);
				for (size_t c = 0; c < _nClasses; ++c) y(c, 0) = y0(data + d,c);
				if (_verbose > 2) {
					Serial.printf ("Batch %d:\n", d);
					Serial.print ("x ="); x.print();
					Serial.print ("y ="); y.print();
				}
// Forward pass
				MLMatrix<float> yhat = forward (x, false);
// Compute error
				err = error (y, yhat);
				totalError += err;
// Backward pass
				backward (yhat, y, d);
			} // end batch
// Update weights
			if (_bestEta) searchEta (x, y, err); // TBD!!!
			else update (batchSize);
			data += batchSize;
			if (_nTrain - data < batchSize) batchSize = _nTrain - data;
		} // end data

		_currError = totalError / _nTrain;
		if (epoch == 1) _firstError = _currError;
		if (_verbose > 1) Serial.printf("Epoch %4d \tAverage error : %8.4f (validation %8.4f)\n", epoch, _currError, _validError);


// Error has decreased...
		if (_currError < _minError or epoch == 1) {
			saveWeights();
			++nimprove;
			_lastBestEpoch = epoch;
			_prevMinError = _minError;
			_minError = _currError;
// Compute error on validation set only if train error decreased
			_validError = testNet(x0, y0, _nTrain, _nValid, false);
			if (!_better) Serial.println();
			Serial.printf("Epoch %4d\tAverage error : %8.4f (validation %8.4f)\n", epoch, _currError, _validError);
			_better = true;
			_nDots = 0;

		} else { // ...error has not decreased
			if (_verbose >0) Serial.print(".");
			_better = false;
			++ _nDots;
		}
// Check if objective has been reached
		if (!_stopTotalError && _currError < stopError) break;
		if (_stopTotalError && _currError + _validError < stopError) break;

// Top-K pruning every 5 epochs
		if (_prune_topk && epoch%5 == 0 && _currError > stopError * 2) { 
			if (pruneTopK(x0, y0) != 0) {
				size();
				float newValidError = testNet(x0, y0, _nTrain, _nValid, false);
				float delta = (newValidError - _validError) / _validError; // relative variation of validation error
				if (delta > 0.1f) { // error changed too much : add a neuron
					int layer = random(1, _nLayers - 1);
					Serial.printf("Validation error changed (%.3f --> %.3f): adding a neuron to layer %d\n", 
						_validError, newValidError, layer);
					addNeuron(layer);
					size();
				}
			}
		}

		++ epoch; // Increase epoch number
		batchSize = _batchSize;

// Shuffle training set after each epoch to break biases
		// if (_nDots > 5 && _shuffleDataset && rand01() < 0.20f) {
		if (_shuffleDataset) {
			if (_verbose > 1) Serial.println ("Shuffling training dataset");
			shuffleDataset (x0, y0, 0, _nTrain);
		}
		
// Prune neurons
		if (_prune_train) {
			// First prune phase (inactive and low activity): error < 4 * _stopError
			if (pruneEpochs == 0 && _currError < 4 * _stopError) {
				++pruneEpochs;
				if (pruneAll()) {
					size(); // Display new network's size
					float newValidError = testNet(x0, y0, _nTrain, _nValid, false);
					float delta = (newValidError - _validError) / _validError;
					if (delta > 0.1f) { // error changed too much : add a neuron
						int layer = random(1, _nLayers - 1);
						Serial.printf("Validation error changed (%.3f --> %.3f): adding a neuron to layer %d\n", 
							_validError, newValidError, layer);
						addNeuron(layer);
						size();
					}
					saveWeights();
				}
			}
			// Second prune phase (inactive neurons): error < 2 * _stopError
			if (pruneEpochs == 1 && _currError < 2 * _stopError) {
				++pruneEpochs;
				if (pruneInactive() != 0) {
					size(); // Display new network's size
					float newValidError = testNet(x0, y0, _nTrain, _nValid, false);
					float delta = (newValidError - _validError) / _validError;
					if (delta > 0.1f) { // error changed too much : add a neuron
						int layer = random(1, _nLayers - 1);
						Serial.printf("Validation error changed (%.3f --> %.3f): adding a neuron to layer %d\n", 
							_validError, newValidError, layer);
						addNeuron(layer);
					}
					saveWeights();
				}
			}
		}
	} // end epochs
	Serial.printf("\nTimer : %.2f s\n", (millis() - chrono) / 1000.0f);
	restoreWeights(); // Use best set of weights

// Evaluate network on test dataset
	float beforeTestError;
	if (_nTest != 0) {
		Serial.printf("\nEvaluation on test data (%d samples):\n",_nTest);
		beforeTestError = testNet(x0, y0, _nTrain + _nValid, _nTest, true);
		Serial.printf("Average test error  : %8.4f\n", beforeTestError);
	}

// Prune network at the end (then re-run test) and save network if required
	bool pruned = false;
	if (_prune_neurons) pruned = pruneAll();
	if (pruned) {
		Serial.printf("\nNew evaluation on test data after pruning:\n",_nTest);
		float testError = testNet(x0, y0, _nTrain + _nValid, _nTest, true);
		Serial.printf("Average test error  : %8.4f\n", testError);
		if (testError <= beforeTestError) saveWeights();
		else Serial.println("Pruning did not improve: pruned network is not saved\n");
	}
}
// End Optimization


// Prediction of result
MLMatrix<float> MLP::predict (MLMatrix<float> x)
{
	if (_verbose > 2) Serial.println("Prediction...");
	MLMatrix<float> yhat;
	for (size_t k = 0; k < _nLayers - 1; ++k) {
		yhat = Weights[k] * x + Biases[k];
		if (_quadLayers) {
			MLMatrix<float> x2 = x.square();
			yhat = yhat + Weights2[k] * x2;
		} //else // No activation for quadratic layers
		yhat = activation(yhat, _activations[k]);
		x = yhat;
	}
	return yhat;
}

// Test the network on the test set
float MLP::testNet(const MLMatrix<float> x0, const MLMatrix<float> y0, const uint16_t begin, const uint16_t number, const bool details)
{
	float error = 0.0f;
	if (_verbose > 1) Serial.println("Test Net...");
	MLMatrix<float> x(_nInputs, 1, 0);
	MLMatrix<float> y(_nClasses, 1, 0);
	MLMatrix<float> errGlob(number, 1, 0);
	MLMatrix<int> Confusion(_nClasses, _nClasses, 0);
	MLMatrix<int> Prec(_nClasses, 1, 0);

	int idh0, idh1, idy0, idy1, correct = 0;
	int nDiff = 0;
	for (size_t i = 0; i < number; ++i) {
		for (size_t j = 0; j < _nInputs; ++j)  x(j,0) = x0(begin + i,j);
		for (size_t j = 0; j < _nClasses; ++j) y(j,0) = y0(begin + i,j); // ground truth
		MLMatrix<float> yhat = predict (x);
		float err = 0.0f;
		if (_enSoftmax) { // If more than one class
			yhat.indexMax(idh0, idh1);
			y.indexMax(idy0, idy1);
			err = abs(idh0 - idy0);
			if (err == 0.0f) ++ correct;
			Confusion(idy0, idh0) += 1;
			Prec(idh0, 0) += 1;

			// Compute the number of times that the difference between 
			// the two first predictions is less than 1/_nClasses
			float valMAx = yhat(idh0,0); // best predicted class
			yhat(idh0,0) = 0.0f;
			yhat.indexMax(idh0, idh1);   // second best predicted class
			float diff = valMAx - yhat(idh0,0); // difference between the two best predicted classes
			if (diff * _nClasses < 1.0f) ++ nDiff; // means that the best prediction has low precision

		} else {
			// Serial.printf("Test %d: expected %f predicted %f (%f)\n",i,y(0,0),yhat(0,0), abs(y(0,0)-yhat(0,0)));
			for (size_t c = 0; c < _nClasses; ++c) 
				err += abs (y(c, 0) - yhat(c, 0));
		}

		error += err;
		errGlob(i,0) = err;
	}

// Print network's statistics
	if (details) {
		Serial.printf(" - Minimum value of error : %8.4f\n", errGlob.min());
		Serial.printf(" - Maximum value of error : %8.4f\n", errGlob.max());
		float mean = errGlob.mean();
		Serial.printf(" - Mean value of error    : %8.4f\n", mean);
		Serial.printf(" - Std deviation of error : %8.4f\n", errGlob.stdev(mean));
		// Serial.printf(" - L0 norm of error : %d\n", errGlob.L0Norm());
		Serial.printf(" - L1 norm of error       : %8.4f\n", errGlob.L1Norm());
		Serial.printf(" - L2 norm of error       : %8.4f\n", errGlob.L2Norm());

		if (_enSoftmax) {
			// Print confusion matrix (rows : true value, cols : predicted value)
			float coef = 100.0f / number;
			Serial.print("Confusion matrix:\nTR/PR");
			for (size_t i = 0; i < _nClasses; ++i) Serial.printf(" %4d", i);
			Serial.println("  (Recall)");
			for (size_t i = 0; i < _nClasses; ++i) {
				Serial.printf ("%3d : ", i);
				int sumI = 0;
				for (size_t j = 0; j < _nClasses; ++j) {
					Serial.printf ("%4d ", Confusion(i,j));
					sumI += Confusion(i,j);
				}
				// Recall
				Serial.printf (" (%5.1f%%)\n",Confusion(i,i) * 100.0f / sumI);
			}
			Serial.print("Prec: ");
			int totConf = 0;
			for (size_t i = 0; i < _nClasses; ++i) {
				totConf += Confusion(i,i);
				if (Prec(i,0) != 0)	Serial.printf ("%4.0f%%", 100.0f * Confusion(i,i) / Prec(i,0));
				else Serial.printf ("%4.0f%%", 0.0f);
			}
			Serial.printf(" ->%5.1f%%\n", 100.0f * totConf / number);
			Serial.printf ("Low precision prediction : %5.1f%%\n", 100.0f * nDiff / number);
		}
	}
	return error / number;
}


// ************************************************************************
//
//     Pruning the network
//
// ************************************************************************
uint16_t MLP::pruneTopK (const MLMatrix<float> x0, const MLMatrix<float> y0)
{
	// Inspired from https://arxiv.org/ftp/arxiv/papers/1711/1711.06528.pdf
	int topK = 0;
	uint16_t before = numberOfWeights();
	float coef = _currError / _stopError / 20;
	if (coef > 1.0f) coef = 1.0f;
	int min_topK = 15 * _nTrain / _batchSize * coef; // Threshold for pruning
	Serial.print("Top K pruning: ");
	for (size_t layer = 1; layer < _nLayers - 1; ++layer) {
		if (_neurons[layer] > _minPerLayer) { // do not prune if not enough neurons
			for (size_t i = 0; i < _neurons[layer]; ++i) {
				if (b_topK[layer][i] < min_topK) { // Candidate for pruning
					/*
							Verify if the performances do not decrease too much after pruning
							compare test score before and after : if less than 10% change ok for pruning
					*/
					float before = testNet(x0, y0, _nTrain + _nValid, _nTest, false);
					MLMatrix<float> tempWRow = Weights[layer - 1].row(i);
					MLMatrix<float> tempWCol = Weights[layer].col(i);
					MLMatrix<float> tempBias = Biases[layer - 1].row(i);
					Weights[layer - 1].setZeroRow(i);
					Weights[layer].setZeroCol(i);
					Biases[layer - 1].setZeroRow(i);
					float after = testNet(x0, y0, _nTrain + _nValid, _nTest, false);
					float score = (after - before) / before;
					// Serial.printf("\n\t\tLayer %d pruning neuron %d: test variation %f%%\n", layer, i, 100*score);
					//
					if (score < 0.1f) {  // OK for pruning
						Serial.printf("\n\tLayer %d pruning neuron %d (score %d / %d)", layer, i, b_topK[layer][i], min_topK);
						removeNeuron(layer, i);
						++topK;
					} else { // Not OK: restore original values
						Weights[layer - 1].setRowMat(i, tempWRow);
						Weights[layer].setColMat(i, tempWCol);
						Biases[layer - 1].setRowMat(i, tempBias);
					}
				}
			}			
		}
	}

	if (topK != 0) {
		float percent = 100.0f - 100.0f * numberOfWeights() / before;
		if (_verbose > 0) Serial.printf("\nTopk: succesfully pruned %d neurons\nNetwork now has %d synapses (-%.2f%%)\n",
			topK, numberOfWeights(), percent);
		saveWeights();
	} else {
		if (_verbose > 0) Serial.println("No candidate neuron found");
	}

	// Reset the counts
	for (size_t layer = 0; layer < _nLayers - 1; ++layer)
		for (size_t i = 0; i < _neurons[i]; ++i) 
			b_topK[layer][i] = 0;
	return topK;
}

uint16_t MLP::pruneInactive()
{
	// Search for neurons with all zero weights (inactive neurons)
	Serial.print("Pruning inactive neurons: ");
	int inact = 0;
	std::vector<int> inacLayers;
	std::vector<int> inacNeurons;
	for (size_t k = 0; k < _nLayers - 2; ++k) {
		MLMatrix<float> W(Weights[k]);
		uint16_t nrows = W.get_rows();
		uint16_t cols = W.get_cols();
		for (size_t i = 0; i < nrows; ++i) {
			uint16_t nZ = W.countZeroRow(i);
			if (nZ == cols) {
				uint16_t nZ2 = cols;
				if (_quadLayers) {
					MLMatrix<float> W2(Weights2[k]);
					nZ2 = W2.countZeroRow(i);
				}
				if (nZ2 == cols) {
					Serial.printf("\n\tLayer %d : neuron %d is inactive", k + 1,i);
					inacLayers.push_back(k);
					inacNeurons.push_back(i);
					++inact;					
				}
			} else {
				if (_neurons[k] > _minPerLayer) { // do not prune if not enough neurons
					float coef = 0.5f;
					float mean = W.meanRow(i);
					if (mean < _zeroThreshold * coef) {
						float mean2 = mean;
						if (_quadLayers) {
							MLMatrix<float> W2(Weights2[k]);
							mean2 += W2.meanRow(i);
						}
						if (mean2 < _zeroThreshold * coef) {
							Serial.printf("\n\tLayer %d : neuron %d is inactive in average (%.4f, %.4f)", k + 1,i, mean2, _zeroThreshold * coef);
							inacLayers.push_back(k);
							inacNeurons.push_back(i);
							++inact;						
						}
					}					
				}
			}
		}
	}

	if (inact == 0) Serial.println ("No inactive neuron found.");
	else {
		Serial.println();
		int count = 0;
		for (size_t i = 0; i < inacLayers.size(); ++i) {
			if ((i > 0) && (inacLayers[i] != inacLayers[i - 1])) count = 0;
			uint16_t layer = inacLayers[i];
			uint16_t neuron = inacNeurons[i] - count;
			++count;
			if (_verbose > 1) Serial.printf("Removing neuron %d of layer %d\n", neuron, layer + 1);
			removeNeuron(layer + 1, neuron);
		}
		if (_verbose > 0) Serial.printf("Succesfully pruned %d inactive neurons\n", inact);
	}
	return inact;
}

uint16_t MLP::pruneLowAct()
{
	// Search for neurons with low activity: more than XXX % weights are zeros in row
	// (set this XXX threshold with setHeurPruning)
	Serial.print("Pruning neurons with low activity: ");
	int lowact = 0;
	std::vector<int> lowacLayers;
	std::vector<int> lowacNeurons;
	for (size_t k = 0; k < _nLayers - 2; ++k) {
		if (_neurons[k] > _minPerLayer) { // do not prune if not enough neurons
			MLMatrix<float> W(Weights[k]);
			uint16_t size = W.get_cols();
			if (size > 6) { // Do not prune thin layers
				for (size_t i = 0; i < W.get_rows(); ++i) {
					uint16_t n = W.countZeroRow(i);
					if (n > int(_pruningThreshold * size)) {
						uint16_t n2 = n;
						if (_quadLayers) {
							MLMatrix<float> W2(Weights2[k]);
							n2 = W2.countZeroRow(i);
						}
						if (n2 > int(_pruningThreshold * size)) {
							Serial.printf("\n\tLayer %d : neuron %d can be pruned (%d)", k + 1, i, n);
							lowacLayers.push_back(k);
							lowacNeurons.push_back(i);
							++lowact;						
						}
					}
				}
			}
		}
	}

	if (lowact == 0) Serial.println ("No low activity neuron found.");
	else {
		Serial.println();
		int count = 0;
		for (size_t i = 0; i < lowacLayers.size(); ++i) {
			if (i > 0 && lowacLayers[i] != lowacLayers[i - 1]) count = 0;
			uint16_t layer = lowacLayers[i];
			uint16_t neuron = lowacNeurons[i] - count;
			++count;
			if (_verbose > 1) Serial.printf("Removing neuron %d of layer %d\n", neuron, layer + 1);
			removeNeuron(layer + 1, neuron);
		}
		if (_verbose > 0) Serial.printf("Succesfully pruned %d low activity neurons\n", lowact);
	}
	return lowact;
}

bool MLP::pruneAll()
{
	Serial.println("Attempting to prune the network");
	uint16_t before = numberOfWeights();
	bool pruned = false;
	uint16_t inact = pruneInactive();
	uint16_t lowact = pruneLowAct();
	uint16_t nPruned = inact + lowact;
	if (nPruned > 0) {
		float percent = 100.0f - 100.0f * numberOfWeights() / before;
		Serial.printf ("Succesfully pruned %d neurons.\nNetwork now has %d synapses (-%.2f%%)\n", 
			nPruned, numberOfWeights(), percent);
		pruned = true;
	}
	// saveWeights();
	return pruned;
}

void MLP::removeNeuron (const int layer, const int number)
{
	if (layer == 0) {
		Serial.println("Prune error: cannot prune input layer !");
		return;
	}
	if (layer == _nLayers - 1) {
		Serial.println("Prune error: cannot prune output layer !");
		return;
	}
	// Weights
	Weights[layer - 1].removeRow(number);
	Biases[layer - 1].removeRow(number);
	Weights[layer].removeCol(number);
	// Gradients
	dWeights[_nLayers - layer - 1].removeRow(number);
	dBiases[_nLayers - layer - 1].removeRow(number);
	dWeights[_nLayers - layer - 2].removeCol(number);
	dWeightsOld[_nLayers - layer - 1].removeRow(number);
	dBiasesOld[_nLayers - layer - 1].removeRow(number);
	dWeightsOld[_nLayers - layer - 2].removeCol(number);
	// Quadratic layer
	if (_quadLayers) {
		Weights2[layer - 1].removeRow(number);
		Weights2[layer].removeCol(number);
		// Gradients
		dWeights2[_nLayers - layer - 1].removeRow(number);
		dWeights2[_nLayers - layer - 2].removeCol(number);
		dWeightsOld2[_nLayers - layer - 1].removeRow(number);
		dWeightsOld2[_nLayers - layer - 2].removeCol(number);
	}
	// Neuron
	--_neurons[layer];
}

void MLP::addNeuron (const int layer)
{
	// Serial.println("Adding neuron is not ok");
	// return;


	if (layer == 0) {
		Serial.println("Cannot add neuron to input layer");
		return;
	}
	if (layer == _nLayers - 1) {
		Serial.println("Cannot add neuron to output layer");
		return;
	}
	// Weights
	Weights[layer - 1].addRow(-0.5f, 0.5f);
	Biases[layer - 1].addRow(-0.5f, 0.5f);
	Weights[layer].addCol(-0.5f, 0.5f);
	// Gradients
	dWeights[_nLayers - layer - 1].addRow(-0.5f, 0.5f);
	dBiases[_nLayers - layer - 1].addRow(-0.5f, 0.5f);
	dWeights[_nLayers - layer - 2].addCol(-0.5f, 0.5f);
	dWeightsOld[_nLayers - layer - 1].addRow(-0.5f, 0.5f);
	dBiasesOld[_nLayers - layer - 1].addRow(-0.5f, 0.5f);
	dWeightsOld[_nLayers - layer - 2].addCol(-0.5f, 0.5f);
	// Quadratic layer
	if (_quadLayers) {
		Weights2[layer - 1].addRow(-0.5f, 0.5f);
		Weights2[layer].addCol(-0.5f, 0.5f);
		// Gradients
		dWeights2[_nLayers - layer - 1].addRow(-0.5f, 0.5f);
		dWeights2[_nLayers - layer - 2].addCol(-0.5f, 0.5f);
		dWeightsOld2[_nLayers - layer - 1].addRow(-0.5f, 0.5f);
		dWeightsOld2[_nLayers - layer - 2].addCol(-0.5f, 0.5f);
	}
	// Neuron
	++_neurons[layer];
}

/******************************************************************************************************************
*******************************************************************************************************************

	Methods for the DeepShift multiplication less network
	https://arxiv.org/abs/1905.13298# 

*******************************************************************************************************************
******************************************************************************************************************/
using DS_t  = int32_t;
using uDS_t = uint32_t;
/**/
// Forward pass
MLMatrix<DS_t> MLP::DSforward (MLMatrix<DS_t> x, bool onlyInference)
{
	if (_verbose > 2) Serial.println("(DeepShift) Forward...");
	float threshold = 0.5f;
	// // 1: Clear workspace
	_aDS.clear(); // sets size to 0
	// _dropoutMasks.clear();
	MLMatrix<DS_t> yhat;
	// 2: Forward pass
	if (!onlyInference) _aDS.push_back(x); // saved for backward pass

	for (size_t k = 0; k < _nLayers - 1; ++k) {
		// P_tilde = round(P)
		MLMatrix<int8_t> Pt = P[k].matRound(0);
		// Stilde = -1, 0, 1 with threshold -0.5 : +0.5
		size_t rows = Pt.get_rows();
		size_t cols = Pt.get_cols();
		MLMatrix<int8_t> St(rows, cols, 0);
		for (size_t i = 0; i < rows; ++i)
	    for (size_t j = 0; j < cols; ++j) {
	    	// St is boolean : 1 for negative, 0 for positive --> sign is (-1)**S or 1-2*S (much faster than pow)
	      if (abs(S[k](i,j)) <= threshold) Pt(i,j) = -32; 
	      if (S[k](i,j) >= threshold) St(i,j) = 0;
	      if (S[k](i,j) < -threshold) St(i,j) = 1;
			}
		// Forward update
 //  Multiply a integer vector x by a shift matrix P and a sign matrix S
 //    y = St . 2**P . x
 //    St is boolean : true (=1) for negative, false (=0) for positive --> sign is (-1)**S or 1-2*S (much faster than pow)
 //    P  is uin8_t  : number of bits to left-shift the value of x
 // NO                   the shift is limited in the range [-10 ; +10] (i.e /1024 to *1024)
 // NO                   to store in unsigned data, the range is shifted by 10, i.e. the real shift is of P-10 with P in [0 ; 20]

		// MLMatrix<int8_t> Dec;
		// Dec = Pt - _shiftP;
		// yhat = (1 - 2 * St) * x.MultShift(Dec);
		yhat = (1 - 2 * St) * x.MultShift(Pt);
		for (size_t i = 0; i < yhat.get_rows(); ++i) {
			yhat(i,0) += DS_Biases[k](i,0);
			yhat(i,0) = (yhat(i,0) < 0) ? 0 : yhat(i,0); // force RELU
		}
		if (!onlyInference) _aDS.push_back(yhat); // saved for backward pass
		x = yhat;
	}
	return yhat;
}

// Back propagation
void MLP::DSbackward (const MLMatrix<DS_t> yhat, const MLMatrix<DS_t> y, const int d)
{
	if (_verbose > 2) Serial.println("(DeepShift) Backward...");
	float ln2 = log(2);
	float threshold = 0.5f;
	MLMatrix<DS_t> delta;
	MLMatrix<DS_t> P_old, S_old;

	for (size_t k = _nLayers - 1; k > 0; --k) {
		if (k == _nLayers - 1) {
			delta  = yhat - y; // last layer (MSE or Cross Entropy)
		} else {
			// delta  = Weights_old.transpose() * delta;
			MLMatrix<int8_t> Pt = P[k].matRound(0);
			// Stilde = -1, 0, 1 with threshold -0.5 : +0.5
			size_t rows = Pt.get_rows();
			size_t cols = Pt.get_cols();
			MLMatrix<int8_t> St(cols, rows, 0);
			for (size_t i = 0; i < rows; ++i)
		    for (size_t j = 0; j < cols; ++j) {
		    	// St is boolean : 1 for negative, 0 for positive --> sign is (-1)**S or 1-2*S (much faster than pow)
		      if (abs(S[k](i,j)) <= threshold) Pt(i,j) = -32; // will right shift of 10 bits
		      if (S[k](i,j) >= threshold) St(j,i) = 0;
		      if (S[k](i,j) < -threshold) St(j,i) = 1;
				}
			MLMatrix<int8_t> Dec;
			// Dec = (Pt - _shiftP).transpose();
			Dec = Pt.transpose();
			delta = (1 - 2 * St) * delta.MultShift(Dec);
		}
		// MLMatrix<uint8_t> b(_aDS[k]);
		for (size_t i = 0; i < _aDS[k].get_rows(); ++i) {
			// b(i,0) = (_aDS[k](i,0) < 0) ? 0 : 1; // Force RELU
			// delta(i,0) = delta(i,0) * b(i,0); 		 // delta = delta.Hadamard(b);
			delta(i,0) = (_aDS[k](i,0) < 0) ? 0 : delta(i,0);
		}

		P_old = P[k - 1];
		S_old = S[k - 1];
		MLMatrix<DS_t> c = delta * _aDS[k - 1].transpose();

		// Compute gradients
		if (d == 0) {
			dP.push_back(c);
			dS.push_back(c);
			DS_dBiases.push_back(delta);
		}	else {
			dP[_nLayers - 1 - k] += c;
			dS[_nLayers - 1 - k] += c;
			dBiases[_nLayers - 1 - k] += delta;
		}
  }
}

void MLP::DSrun(MLMatrix<float> x0, MLMatrix<float> y0, int maxEpochs, int batchSize, float stopError)
{
	if (!_changeBSize) Serial.printf("Batch size = %d\n",batchSize);
	Serial.printf("Stopping if error < %.3f\n",stopError);
	Serial.println("\nRunning optimization...");

// Prepare dataset
	if (!_datasetSplit) {
		_nTrain = _nData * rTrain;
		_nValid = _nData * rValid;
		_nTest  = _nData * rTest;
		Serial.printf ("Dataset split in: %d train + %d validation + %d test data\n", _nTrain, _nValid, _nTest);
		_datasetSplit = true;
	}
	// Shuffle dataset
	shuffleDataset (x0, y0, 0, _nData);

	// Pre-train on a subset of the dataset (20%) for fastest beginning
	uint16_t savenTrain, savenValid, nimprove = 0;
	bool subTrain;
	if (_dataSubset) {
		savenTrain = _nTrain;
		savenValid = _nValid;
		_nTrain /= 5;
		_nValid /= 5;
		subTrain = true;
	}

// Prepare topK pruning
	if (_prune_topk) {
		b_topK.resize(_nLayers - 1);
		for (size_t i = 0; i < _nLayers - 1; ++i) b_topK[i].resize(_neurons[i], 0.0f);
	}

// Initialize weights
	if (_initialize) {
		if (_verbose > 0) Serial.println("Creating a new network");
		// (_selectWeights) ? searchBestWeights(x0, y0) : DSinitWeights();
		DSinitWeights(-5, 6);
	}

// Estimate training duration
	uint32_t duration = estimateDuration (maxEpochs);
	Serial.printf("Estimated maximum duration : %.2f s for %d epochs\n", duration/1000.0f, maxEpochs);

// Variables
	MLMatrix<DS_t> x(_nInputs,  1, 0.0f);  // Input array
	MLMatrix<DS_t> y(_nClasses, 1, 0.0f);  // Ground truth
	bool _better = false;
	_batchSize = batchSize;
	if (_batchSize >= _nTrain) _batchSize = max(1, _nTrain / 5);
	_stopError = stopError;
	_firstRun = true;
	_maxEpochs = maxEpochs;
	byte pruneEpochs = 0;

	// Epochs loop...
	int epoch = 1;
	unsigned long chrono = millis();
	while (epoch <= _maxEpochs) {
		if (_verbose > 1) Serial.printf("Epoch %d\n", epoch);

// Check if pre-train phase is finished
		if (_dataSubset && subTrain) {
			if (epoch > _maxEpochs / 3 || nimprove > 4) { // Back to entire dataset
				_nTrain = savenTrain;
				_nValid = savenValid;
				subTrain = false;
				_minError *= 10.0f;
				Serial.println("Now training on the entire dataset");
			}
		}

// Manage hyper-parameters
		if (epoch > 1) heuristics(epoch, _maxEpochs, _better);

		_prevError = _currError;
		float totalError = 0.0f;
		int data = 0;

		if (_changeBSize) { // (heuristics) change batch size
			float coef = _maxBS + epoch * (_minBS - _maxBS) / _maxEpochs;
			batchSize = int(coef * _nTrain);
			if (batchSize < 1) batchSize = 1;
			if (_verbose > 0) Serial.printf ("Batch size = %d\n", batchSize);
		}
// Loop over training set
		while (data < _nTrain) {
			dWeightsOld = dWeights;
			if (_quadLayers) dWeightsOld2 = dWeights2;
			dBiasesOld = dBiases;
			dWeights.clear();
			if (_quadLayers) dWeights2.clear();
			dBiases.clear();
			if (_verbose > 2) Serial.printf ("Data number %d (%d)\n", data, batchSize);
			uDS_t err;

// Loop over minibatch
			float coef = pow(2.0f, 16);
			for (size_t d = 0; d < batchSize; ++d) {
				for (size_t i = 0; i < _nInputs; ++i)  x(i, 0) = x0(data + d,i) * coef;
				for (size_t c = 0; c < _nClasses; ++c) y(c, 0) = y0(data + d,c) * coef;
				if (_verbose > 2) {
					Serial.printf ("Batch %d:\n", d);
					Serial.print ("x ="); x.print();
					Serial.print ("y ="); y.print();
				}
// Forward pass
				MLMatrix<DS_t> yhat = DSforward (x, false);
// Compute error
				err = DSerror (y, yhat);
				totalError += err / coef;
// Backward pass
				DSbackward (yhat, y, d);
			} // end batch
// Update weights
			// if (_bestEta) searchEta (x, y, err); // TBD!!!
			// else 
			DSupdate (batchSize);
			data += batchSize;
			if (_nTrain - data < batchSize) batchSize = _nTrain - data;
		} // end data

		_currError = totalError / _nTrain;
		if (epoch == 1) _firstError = _currError;
		if (_verbose > 1) Serial.printf("Epoch %4d \tAverage error : %8.4f (validation %8.4f)\n", epoch, _currError, _validError);


// Error has decreased...
		if (_currError < _minError or epoch == 1) {
			saveWeights();
			++nimprove;
			_lastBestEpoch = epoch;
			_prevMinError = _minError;
			_minError = _currError;
// Compute error on validation set only if train error decreased
			_validError = testNet(x0, y0, _nTrain, _nValid, false);
			if (!_better) Serial.println();
			Serial.printf("Epoch %4d\tAverage error : %8.4f (validation %8.4f)\n", epoch, _currError, _validError);
			_better = true;
			_nDots = 0;

		} else { // ...error has not decreased
			if (_verbose >0) Serial.print(".");
			_better = false;
			++ _nDots;
		}
// Check if objective has been reached
		if (!_stopTotalError && _currError < stopError) break;
		if (_stopTotalError && _currError + _validError < stopError) break;

// Top-K pruning every 5 epochs
		if (_prune_topk && epoch%5 == 0 && _currError > stopError * 2) { 
			if (pruneTopK(x0, y0) != 0) size();
		}

		++ epoch; // Increase epoch number
		batchSize = _batchSize;

// Shuffle training set after each epoch to break biases
		// if (_nDots > 5 && _shuffleDataset && rand01() < 0.20f) {
		if (_shuffleDataset) {
			if (_verbose > 1) Serial.println ("Shuffling training dataset");
			shuffleDataset (x0, y0, 0, _nTrain);
		}
		
// Prune neurons
		if (_prune_train) {
			// First prune phase (inactive and low activity): error < 4 * _stopError
			if (pruneEpochs == 0 && _currError < 4 * _stopError) {
				++pruneEpochs;
				if (pruneAll()) size(); // Display new network's size
			}
			// Second prune phase (inactive neurons): error < 2 * _stopError
			if (pruneEpochs == 1 && _currError < 2 * _stopError) {
				++pruneEpochs;
				if (pruneInactive() != 0) size(); // Display new network's size
			}
		}
	} // end epochs
	Serial.printf("\nTimer : %.2f s\n", (millis() - chrono) / 1000.0f);
	restoreWeights(); // Use best set of weights

// Evaluate network on test dataset
	float beforeTestError;
	if (_nTest != 0) {
		Serial.printf("\nEvaluation on test data (%d samples):\n",_nTest);
		beforeTestError = testNet(x0, y0, _nTrain + _nValid, _nTest, true);
		Serial.printf("Average test error  : %8.4f\n", beforeTestError);
	}

// Prune network at the end (then re-run test) and save network if required
	bool pruned = false;
	if (_prune_neurons) pruned = pruneAll();
	if (pruned) {
		Serial.printf("\nNew evaluation on test data after pruning:\n",_nTest);
		float testError = testNet(x0, y0, _nTrain + _nValid, _nTest, true);
		Serial.printf("Average test error  : %8.4f\n", testError);
		if (testError <= beforeTestError) saveWeights();
		else Serial.println("Pruning did not improve: pruned network is not saved");
	}
}
// End Optimization

// Initialize weights & biases : random uniform or Xavier
void MLP::DSinitWeights (const DS_t vmin, const DS_t vmax)
{
	if (_verbose > 2) Serial.println("(DeepShift) Init weights...");
	float coef = pow(2.0f, 15);
	P.clear();
	S.clear();
	DS_Biases.clear();
	if (_quadLayers) { P2.clear(); S2.clear(); }

	for (size_t k = 0; k < _nLayers - 1; ++k) {
		MLMatrix<DS_t> MP(_neurons[k + 1], _neurons[k], vmin, vmax);
		MLMatrix<DS_t> MS(_neurons[k + 1], _neurons[k], 0, 2);
		MLMatrix<DS_t> B(_neurons[k + 1], 1, -coef, coef);

		P.push_back(MP);
		S.push_back(MS);
		DS_Biases.push_back(B);

		if (_quadLayers) {
			MLMatrix<DS_t> MP2(_neurons[k + 1], _neurons[k], vmin, vmax);
			MLMatrix<DS_t> MS2(_neurons[k + 1], _neurons[k], 0, 2);
			P2.push_back(MP2);
			S2.push_back(MS2);
		}

		if (_verbose > 2) {
			MP.print(); MS.print();
			if (_quadLayers) {
				MLMatrix<DS_t> MP2 = P2.back();
				MP2.print();
				MLMatrix<DS_t> MS2 = S2.back();
				MS2.print();
			}
			B.print();
		}
	}
}


uDS_t MLP::DSerror (MLMatrix<DS_t> y, MLMatrix<DS_t> yhat) const
{
	if (_verbose > 2) Serial.print("Error... ");
	float coef = pow(2.0f, 16);
	DS_t err = 0.0f;
	int idh0, idh1, idy0, idy1;
	if (_enSoftmax) {
		yhat.indexMax(idh0, idh1);
		y.indexMax(idy0, idy1);
		err = abs(idh0 - idy0) * coef;
	} else {
		for (size_t c = 0; c < _nClasses; ++c) 
			err += abs (y(c, 0) - yhat(c, 0));
	}
	return err;
}

// Update weights & biases
void MLP::DSupdate (const int batchSize)
{
	float eta = _eta;
	for (size_t k = _nLayers - 1; k > 0; --k) {
		// Testing layer specific learning rates (aka differential LR): not convincing
		// eta = _eta * (1.0f + (_nLayers - k - 1.0f) / (_nLayers - 2.0f)); // LR decreasing with deeper layers
		// eta = _eta * (1.0f + (k - 1.0f) / (_nLayers - 2.0f)); // LR increasing with deeper layers (better)
		MLMatrix<DS_t> MP = P[k - 1];
		MLMatrix<DS_t> MS = S[k - 1];
		// MLMatrix<float> B = Biases[k - 1];

// Gradient clipping
		if (_gradClip && !_firstRun) {
			DS_dBiases[_nLayers - 1 - k].clipMax(_gradClipValue);
			dP[_nLayers - 1 - k].clipMax(_gradClipValue);
			if (_quadLayers) dP2[_nLayers - 1 - k].clipMax(_gradClipValue);
		}

// Rescale gradient
		if (_gradScaling && !_firstRun) {
			bool zeroNorm = dBiases[_nLayers - 1 - k].normScale2(_gradScale);
			if (zeroNorm) {
				Serial.printf("Warning: biases gradients are zero (layer %d)\n", k);
				DS_dBiases[_nLayers - 1 - k].print();
			}
			zeroNorm = dP[_nLayers - 1 - k].normScale2(_gradScale);
			if (_quadLayers) zeroNorm += dP2[_nLayers - 1 - k].normScale2(_gradScale);
			if (zeroNorm)	{
				Serial.printf("Warning: weights gradients are zero (layer %d)\n", k);
				dP[_nLayers - 1 - k].print();
			}
		}

// Update weights
		DS_Biases[k - 1] = DS_Biases[k - 1] - DS_dBiases[_nLayers - 1 - k]  * (eta / batchSize);
		P[k - 1] = P[k - 1] - dP[_nLayers - 1 - k] * (eta / batchSize);
		S[k - 1] = S[k - 1] - dS[_nLayers - 1 - k] * (eta / batchSize);
		if (_quadLayers) {
			P2[k - 1] = P2[k - 1] - dP2[_nLayers - 1 - k] * (eta / batchSize);
			S2[k - 1] = S2[k - 1] - dS2[_nLayers - 1 - k] * (eta / batchSize);
		}
		if (!_firstRun) {
			DS_Biases[k - 1] = DS_Biases[k - 1] - DS_dBiasesOld[_nLayers - 1 - k]  * (_momentum / batchSize);
			P[k - 1] = P[k - 1] - dPOld[_nLayers - 1 - k] * (_momentum / batchSize);
			S[k - 1] = S[k - 1] - dSOld[_nLayers - 1 - k] * (_momentum / batchSize);
			if (_quadLayers) {
				P2[k - 1] = P2[k - 1] - dP2Old[_nLayers - 1 - k] * (_momentum / batchSize);
				S2[k - 1] = S2[k - 1] - dS2Old[_nLayers - 1 - k] * (_momentum / batchSize);
			}
		}

// Apply regularization penalty
		if (_regulL1)  {
			// DS_Biases[k - 1]  -= B.sgn() * _lambdaRegulL1 * (eta / batchSize);
			P[k - 1] = P[k - 1] - MP.sgn() * _lambdaRegulL1 * (eta / batchSize);
		}
		if (_regulL2)  {
			// DS_Biases[k - 1]  -= B * _lambdaRegulL2 * (eta / batchSize);
			P[k - 1] = P[k - 1] - MP * _lambdaRegulL2 * (eta / batchSize);
		}

// Force weights to zero if lower than threshold
		// if (_zeroWeights) {
		// 	int nbBiasClip    = DS_Biases[k - 1].clipToZero(_zeroThreshold);
		// 	int nbWeightClip  = P[k - 1].clipMin(-6);
		// 	int nbWeight2Clip = (_quadLayers) ? P2[k - 1].clipToZero(_zeroThreshold) : 0;
		// 	if (nbBiasClip + nbWeightClip + nbWeight2Clip == 0 && _verbose > 1) 
		// 		Serial.printf ("Warning: clipping threshold (%.4f) too low, no weights to clip...\n", _zeroThreshold);
		// }
	}
	_firstRun = false;
}
