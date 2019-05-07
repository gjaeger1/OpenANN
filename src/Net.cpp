#include <OpenANN/Net.h>
#include <OpenANN/layers/Input.h>
#include <OpenANN/layers/AlphaBetaFilter.h>
#include <OpenANN/layers/FullyConnected.h>
#include <OpenANN/layers/RBF.h>
#include <OpenANN/layers/Compressed.h>
#include <OpenANN/layers/Extreme.h>
#include <OpenANN/layers/Convolutional.h>
#include <OpenANN/layers/Subsampling.h>
#include <OpenANN/layers/MaxPooling.h>
#include <OpenANN/layers/LocalResponseNormalization.h>
#include <OpenANN/layers/Dropout.h>
#include <OpenANN/RBM.h>
#include <OpenANN/SparseAutoEncoder.h>
#include <OpenANN/IntrinsicPlasticity.h>
#include <OpenANN/ErrorFunctions.h>
#include <OpenANN/io/DirectStorageDataSet.h>
#include <OpenANN/util/OpenANNException.h>
#include <OpenANN/util/AssertionMacros.h>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace OpenANN
{

Net::Net()
  : errorFunction(MSE), dropout(false), backpropToAll(false), initialized(false), P(-1), L(0)
{
  layers.reserve(3);
  infos.reserve(3);
  architecture.precision(20);
}

Net::~Net()
{
  this->clearLayers();
}

void Net::clearLayers()
{
  for(int i = 0; i < layers.size(); i++)
  {
    delete layers[i];
    layers[i] = 0;
  }
  layers.clear();
  infos.clear();
  parameters.clear();
  derivatives.clear();
}

/**
* @brief Copy Constructor
*/
Net::Net(const Net& other) : Learner(other), errorFunction(MSE), dropout(false), backpropToAll(false), initialized(false), P(-1), L(0)
{
    OPENANN_DEBUG << "Copy Constructor\n";
    OPENANN_DEBUG << "Before: Other Architecture:\n" << other.architecture.str() << "\n Own Architecture:\n" << this->architecture.str() << "\n";
    architecture.precision(20);
    //std::cout << "Using Copy Constructor\n";
    if(this == &other)
        return;

    this->regularization = other.regularization;
    this->errorFunction = other.errorFunction;
    this->dropout = other.dropout;
    this->backpropToAll = other.backpropToAll;
    this->L = 0; // will be set when loading from string stream
    this->initialized = false;
    this->tempGradient = other.tempGradient;
    this->tempInput = other.tempInput;
    this->tempOutput = other.tempOutput;
    this->tempError = other.tempError;

    // copy layers by load/saving them to stringstream
    std::stringstream ss;
    ss.precision(20);
    ss.str(other.architecture.str());
    OPENANN_DEBUG << "Copied Architecture:\n" << ss.str() << "\n";
    this->load(ss);
    this->setParameters(other.parameterVector);
    OPENANN_DEBUG << "After: Other Architecture:\n" << other.architecture.str() << "\n Own Architecture:\n" << this->architecture.str() << "\n";
}

/**
* @brief Move Contstructor
*/
Net::Net(Net&& other)  : Learner(other), errorFunction(MSE), dropout(false), backpropToAll(false), initialized(false), P(-1), L(0)
{
    OPENANN_DEBUG << "Move Constructor\n";
    OPENANN_DEBUG << "Before: Other Architecture:\n" << other.architecture.str() << "\n Own Architecture:\n" << this->architecture.str() << "\n";
    //std::cout << "Using Move Constructor\n";
    architecture.precision(20);
    std::swap(this->architecture, other.architecture);
    std::swap(this->infos,other.infos);
    std::swap(this->regularization,other.regularization);
    std::swap(this->errorFunction , other.errorFunction);
    std::swap(this->dropout , other.dropout);
    std::swap(this->backpropToAll, other.backpropToAll);
    std::swap(this->initialized , other.initialized);
    std::swap(this->P, other.P);
    std::swap(this->L, other.L);
    std::swap(this->parameterVector, other.parameterVector);
    std::swap(this->tempGradient, other.tempGradient);
    std::swap(this->tempInput, other.tempInput);
    std::swap(this->tempOutput, other.tempOutput);
    std::swap(this->tempError ,other.tempError);

    this->layers = std::move(other.layers);
    this->parameters = std::move(other.parameters);
    this->derivatives = std::move(other.derivatives);
    OPENANN_DEBUG << "After: Other Architecture:\n" << other.architecture.str() << "\n Own Architecture:\n" << this->architecture.str() << "\n";
}

/**
* @brief Copy Assignment Operator
*/
Net& Net::operator=(const Net& other)
{
    OPENANN_DEBUG << "Copy Assignment Operator\n";
    OPENANN_DEBUG << "Before: Other Architecture:\n" << other.architecture.str() << "\n Own Architecture:\n" << this->architecture.str() << "\n";
    //std::cout << "Using Copy Assignment Operator\n";
    if(this == &other)
        return *this;

    Learner::operator =(other);

    this->clearLayers();
    this->architecture.str(std::string());

    this->regularization = other.regularization;
    this->errorFunction = other.errorFunction;
    this->dropout = other.dropout;
    this->backpropToAll = other.backpropToAll;
    this->initialized = false;
    this->P = -1;
    this->L = 0; // will be set when loading from string stream
    this->tempGradient = other.tempGradient;
    this->tempInput = other.tempInput;
    this->tempOutput = other.tempOutput;
    this->tempError = other.tempError;


    // copy layers by load/saving them to stringstream
    std::stringstream ss;
    ss.precision(20);
    ss << other.architecture.str();
    this->load(ss);
    this->setParameters(other.parameterVector);
    OPENANN_DEBUG << "After: Other Architecture:\n" << other.architecture.str() << "\n Own Architecture:\n" << this->architecture.str() << "\n";
    return *this;
}

/**
* @brief Move Assignment Operator
*/
Net& Net::operator=(Net&& other)
{
    OPENANN_DEBUG << "Move Assignment Operator\n";
    OPENANN_DEBUG << "Before: Other Architecture:\n" << other.architecture.str() << "\n Own Architecture:\n" << this->architecture.str() << "\n";
    //std::cout << "Using Move Assignment Operator\n";

    if(this == &other)
        return *this;

    this->clearLayers();
    this->architecture.str(std::string());

    Learner::operator =(other);
    std::swap(this->architecture, other.architecture);
    std::swap(this->infos,other.infos);
    std::swap(this->regularization,other.regularization);
    std::swap(this->errorFunction , other.errorFunction);
    std::swap(this->dropout , other.dropout);
    std::swap(this->backpropToAll, other.backpropToAll);
    std::swap(this->initialized , other.initialized);
    std::swap(this->P, other.P);
    std::swap(this->L, other.L);
    std::swap(this->parameterVector, other.parameterVector);
    std::swap(this->tempGradient, other.tempGradient);
    std::swap(this->tempInput, other.tempInput);
    std::swap(this->tempOutput, other.tempOutput);
    std::swap(this->tempError ,other.tempError);

    this->layers = std::move(other.layers);
    this->parameters = std::move(other.parameters);
    this->derivatives = std::move(other.derivatives);

    OPENANN_DEBUG << "After: Other Architecture:\n" << other.architecture.str() << "\n Own Architecture:\n" << this->architecture.str() << "\n";

    return *this;
}

Net& Net::inputLayer(int dim1, int dim2, int dim3)
{
  architecture << "input " << dim1 << " " << dim2 << " " << dim3 << " ";
  return addLayer(new Input(dim1, dim2, dim3));
}

Net& Net::alphaBetaFilterLayer(double deltaT, double stdDev)
{
  architecture << "alpha_beta_filter " << deltaT << " " << stdDev << " ";
  return addLayer(new AlphaBetaFilter(infos.back(), deltaT, stdDev));
}

Net& Net::fullyConnectedLayer(int units, ActivationFunction act, double stdDev,
                              bool bias)
{
  architecture << "fully_connected " << units << " " << (int) act << " "
      << stdDev << " " << bias << " ";
  return addLayer(new FullyConnected(infos.back(), units, bias, act, stdDev,
                                     regularization));
}

Net& Net::rbfLayer(int units, double stdDev)
{
  architecture << "rbf " << units << " "  << stdDev << " ";
  return addLayer(new RBF(infos.back(), units, stdDev, regularization));
}

Net& Net::restrictedBoltzmannMachineLayer(int H, int cdN, double stdDev,
                                          bool backprop)
{
  architecture << "rbm " << H << " " << cdN << " " << stdDev << " "
      << backprop << " ";
  return addLayer(new RBM(infos.back().outputs(), H, cdN, stdDev,
                          backprop, regularization));
}

Net& Net::sparseAutoEncoderLayer(int H, double beta, double rho,
                                 ActivationFunction act)
{
  architecture << "sae " << H << " " << beta << " " << rho << " " << (int) act
      << " ";
  return addLayer(new SparseAutoEncoder(infos.back().outputs(), H, beta, rho,
                                        regularization.l2Penalty, act));
}

Net& Net::compressedLayer(int units, int params, ActivationFunction act,
                          const std::string& compression, double stdDev,
                          bool bias)
{
  architecture << "compressed " << units << " " << params << " " << (int) act
      << " " << compression << " " << stdDev << " " << bias << " ";
  return addLayer(new Compressed(infos.back(), units, params, bias, act,
                                 compression, stdDev, regularization));
}

Net& Net::extremeLayer(int units, ActivationFunction act, double stdDev,
                       bool bias)
{
  architecture << "extreme " << units << " " << (int) act << " " << stdDev
      << " " << bias << " ";
  return addLayer(new Extreme(infos.back(), units, bias, act, stdDev));
}

Net& Net::intrinsicPlasticityLayer(double targetMean, double stdDev)
{
  architecture << "intrinsic_plasticity " << targetMean << " " << stdDev << " ";
  return addLayer(new IntrinsicPlasticity(infos.back().outputs(), targetMean,
                                          stdDev));
}

Net& Net::convolutionalLayer(int featureMaps, int kernelRows, int kernelCols,
                             ActivationFunction act, double stdDev, bool bias)
{
  architecture << "convolutional " << featureMaps << " " << kernelRows << " "
      << kernelCols << " " << (int) act << " " << stdDev << " " << bias << " ";
  return addLayer(new Convolutional(infos.back(), featureMaps, kernelRows,
                                    kernelCols, bias, act, stdDev, regularization));
}

Net& Net::subsamplingLayer(int kernelRows, int kernelCols,
                           ActivationFunction act, double stdDev, bool bias)
{
  architecture << "subsampling " << kernelRows << " " << kernelCols << " "
      << (int) act << " " << stdDev << " " << bias << " ";
  return addLayer(new Subsampling(infos.back(), kernelRows, kernelCols, bias,
                                  act, stdDev, regularization));
}

Net& Net::maxPoolingLayer(int kernelRows, int kernelCols)
{
  architecture << "max_pooling " << kernelRows << " " << kernelCols << " ";
  return addLayer(new MaxPooling(infos.back(), kernelRows, kernelCols));
}

Net& Net::localReponseNormalizationLayer(double k, int n, double alpha,
                                         double beta)
{
  architecture << "local_response_normalization " << k << " " << n << " "
      << alpha << " " << beta << " ";
  return addLayer(new LocalResponseNormalization(infos.back(), k, n, alpha,
                                                 beta));
}

Net& Net::dropoutLayer(double dropoutProbability)
{
  architecture << "dropout " << dropoutProbability << " ";
  return addLayer(new Dropout(infos.back(), dropoutProbability));
}

Net& Net::addLayer(Layer* layer)
{
  OPENANN_CHECK(layer != 0);

  OutputInfo info = layer->initialize(parameters, derivatives);
  layers.push_back(layer);
  infos.push_back(info);
  L++;
  return *this;
}

Net& Net::addOutputLayer(Layer* layer)
{
  addLayer(layer);
  initializeNetwork();
  return *this;
}


Net& Net::outputLayer(int units, ActivationFunction act, double stdDev, bool bias)
{
  architecture << "output " << units << " " << (int) act << " " << stdDev
      << " " << bias << " ";
  addLayer(new FullyConnected(infos.back(), units, bias, act, stdDev,
                              regularization));
  initializeNetwork();
  return *this;
}

Net& Net::compressedOutputLayer(int units, int params, ActivationFunction act,
                                const std::string& compression, double stdDev,
                                bool bias)
{
  architecture << "compressed_output " << units << " " << params << " "
      << (int) act << " " << compression << " " << stdDev << " " << bias << " ";
  addLayer(new Compressed(infos.back(), units, params, bias, act, compression,
                          stdDev, regularization));
  initializeNetwork();
  return *this;
}

unsigned int Net::numberOflayers() const
{
  return L;
}

Layer& Net::getLayer(unsigned int l)
{
  OPENANN_CHECK(l >= 0 && l < L);
  return *layers[l];
}

OutputInfo Net::getOutputInfo(unsigned int l) const
{
  OPENANN_CHECK(l >= 0 && l < L);
  return infos[l];
}

DataSet* Net::propagateDataSet(DataSet& dataSet, int l)
{
  Eigen::MatrixXd X(dataSet.samples(), dataSet.inputs());
  Eigen::MatrixXd T(dataSet.samples(), dataSet.outputs());
  for(int n = 0; n < dataSet.samples(); n++)
  {
    tempInput = dataSet.getInstance(n).transpose();
    Eigen::MatrixXd* y = &tempInput;
    int i = 0;
    for(std::vector<Layer*>::iterator layer = layers.begin();
        layer != layers.end() && i < l; ++layer)
      (**layer).forwardPropagate(y, y, dropout);
    tempOutput = *y;
    X.row(n) = tempOutput;
    T.row(n) = dataSet.getTarget(n).transpose();
  }
  DirectStorageDataSet* transformedDataSet = new DirectStorageDataSet(&X, &T);
  return transformedDataSet;
}

void Net::save(const std::string& fileName)  const
{
  std::ofstream file(fileName.c_str());
  if(!file.is_open())
    throw OpenANNException("Could not open '" + fileName + "'.'");
  save(file);
  file.close();
}

void Net::save(std::ostream& stream)  const
{
  stream << architecture.str() << "parameters " << currentParameters();
}

void Net::load(const std::string& fileName)
{
  std::ifstream file(fileName.c_str());
  if(!file.is_open())
    throw OpenANNException("Could not open '" + fileName + "'.'");
  load(file);
  file.close();
}

void Net::load(std::istream& stream)
{
  OPENANN_DEBUG << "Net::load\n";

  while(!stream.eof())
  {
    std::string type;
    stream >> type;

    if(type == "")
        return;

    if(type == "input")
    {
      int dim1, dim2, dim3;
      stream >> dim1 >> dim2 >> dim3;
      OPENANN_DEBUG << "input " << dim1 << " " << dim2 << " " << dim3;
      inputLayer(dim1, dim2, dim3);
    }
    else if(type == "alpha_beta_filter")
    {
      double deltaT, stdDev;
      stream >> deltaT >> stdDev;
      OPENANN_DEBUG << "alpha_beta_filter" << deltaT << " " << stdDev;
      alphaBetaFilterLayer(deltaT, stdDev);
    }
    else if(type == "fully_connected")
    {
      int units;
      int act;
      double stdDev;
      bool bias;
      stream >> units >> act >> stdDev >> bias;
      OPENANN_DEBUG << "fully_connected " << units << " " << act << " "
          << stdDev << " " << bias;
      fullyConnectedLayer(units, (ActivationFunction) act, stdDev, bias);
    }
    else if(type == "rbf")
    {
      int units;
      double stdDev;
      stream >> units >> stdDev;
      OPENANN_DEBUG << "rbf " << units << " " << stdDev;
      rbfLayer(units, stdDev);
    }
    else if(type == "rbm")
    {
      int H;
      int cdN;
      double stdDev;
      bool backprop;
      stream >> H >> cdN >> stdDev >> backprop;
      OPENANN_DEBUG << "rbm " << H << " " << cdN << " " << stdDev << " "
          << backprop;
      restrictedBoltzmannMachineLayer(H, cdN, stdDev, backprop);
    }
    else if(type == "sae")
    {
      int H;
      double beta, rho;
      int act;
      stream >> H >> beta >> rho >> act;
      OPENANN_DEBUG << "sae " << H << " " << beta << " " << rho << " " << act;
      sparseAutoEncoderLayer(H, beta, rho, (ActivationFunction) act);
    }
    else if(type == "compressed")
    {
      int units;
      int params;
      int act;
      std::string compression;
      double stdDev;
      bool bias;
      stream >> units >> params >> act >> compression >> stdDev >> bias;
      OPENANN_DEBUG << "compressed " << units << " " << params << " " << act
          << " " << compression << " " << stdDev << " " << bias;
      compressedLayer(units, params, (ActivationFunction) act, compression,
                      stdDev, bias);
    }
    else if(type == "extreme")
    {
      int units;
      int act;
      double stdDev;
      bool bias;
      stream >> units >> act >> stdDev >> bias;
      OPENANN_DEBUG << "extreme " << units << " " << act << " " << stdDev
          << " " << bias;
      extremeLayer(units, (ActivationFunction) act, stdDev, bias);
    }
    else if(type == "intrinsic_plasticity")
    {
      double targetMean;
      double stdDev;
      stream >> targetMean >> stdDev;
      OPENANN_DEBUG << "intrinsic_plasticity " << targetMean << " " << stdDev;
      intrinsicPlasticityLayer(targetMean, stdDev);
    }
    else if(type == "convolutional")
    {
      int featureMaps, kernelRows, kernelCols, act;
      double stdDev;
      bool bias;
      stream >> featureMaps >> kernelRows >> kernelCols >> act >> stdDev >> bias;
      OPENANN_DEBUG << "convolutional " << featureMaps << " " << kernelRows
          << " " << kernelCols << " " << act << " " << stdDev << " " << bias;
      convolutionalLayer(featureMaps, kernelRows, kernelCols,
                         (ActivationFunction) act, stdDev, bias);
    }
    else if(type == "subsampling")
    {
      int kernelRows, kernelCols, act;
      double stdDev;
      bool bias;
      stream >> kernelRows >> kernelCols >> act >> stdDev >> bias;
      OPENANN_DEBUG << "subsampling " << kernelRows << " " << kernelCols
          << " " << act << " " << stdDev << " " << bias;
      subsamplingLayer(kernelRows, kernelCols, (ActivationFunction) act,
                       stdDev, bias);
    }
    else if(type == "max_pooling")
    {
      int kernelRows, kernelCols;
      stream >> kernelRows >> kernelCols;
      OPENANN_DEBUG << "max_pooling " << kernelRows << " " << kernelCols;
      maxPoolingLayer(kernelRows, kernelCols);
    }
    else if(type == "local_response_normalization")
    {
      double k, alpha, beta;
      int n;
      stream >> k >> n >> alpha >> beta;
      OPENANN_DEBUG << "local_response_normalization " << k << " " << n << " "
          << alpha << " " << beta;
      localReponseNormalizationLayer(k, n, alpha, beta);
    }
    else if(type == "dropout")
    {
      double dropoutProbability;
      stream >> dropoutProbability;
      OPENANN_DEBUG << "dropout " << dropoutProbability;
      dropoutLayer(dropoutProbability);
    }
    else if(type == "output")
    {
      int units;
      int act;
      double stdDev;
      bool bias;
      stream >> units >> act >> stdDev >> bias;
      OPENANN_DEBUG << "output " << units << " " << act << " " << stdDev
          << " " << bias;
      outputLayer(units, (ActivationFunction) act, stdDev, bias);
    }
    else if(type == "compressed_output")
    {
      int units;
      int params;
      int act;
      std::string compression;
      double stdDev;
      bool bias;
      stream >> units >> params >> act >> compression >> stdDev >> bias;
      OPENANN_DEBUG << "compressed_output " << units << " " << params << " "
          << act << " " << compression << " " << stdDev << " " << bias;
      compressedOutputLayer(units, params, (ActivationFunction) act,
                            compression, stdDev, bias);
    }
    else if(type == "error_function")
    {
      int errorFunction;
      stream >> errorFunction;
      OPENANN_DEBUG << "error_function " << errorFunction;
      setErrorFunction((ErrorFunction) errorFunction);
    }
    else if(type == "regularization")
    {
      double l1Penalty, l2Penalty, maxSquaredWeightNorm;
      stream >> l1Penalty >> l2Penalty >> maxSquaredWeightNorm;
      OPENANN_DEBUG << "regularization " << l1Penalty << " " << l2Penalty
          << " " << maxSquaredWeightNorm;
      setRegularization(l1Penalty, l2Penalty, maxSquaredWeightNorm);
    }
    else if(type == "parameters")
    {
      double p = 0.0;
      for(int i = 0; i < dimension(); i++)
        stream >> parameterVector(i);
      setParameters(parameterVector);
    }
    else
    {
      if(stream.eof())
      {
        OPENANN_INFO << "Unknown layer type '" << type << "' found at end of stream. Ignoring...";
        return;
      }
      else
      {
        throw OpenANNException("Unknown layer type: '" + type + "'.");
      }
    }
  }
}

Net& Net::useDropout(bool activate)
{
  dropout = activate;
  return *this;
}

Net& Net::backpropagateThroughAllLayers(bool activate)
{
  this->backpropToAll = activate;
  return *this;
}

Net& Net::setRegularization(double l1Penalty, double l2Penalty,
                            double maxSquaredWeightNorm)
{
  architecture << "regularization " << l1Penalty << " " << l2Penalty << " "
      << maxSquaredWeightNorm << " ";
  regularization.l1Penalty = l1Penalty;
  regularization.l2Penalty = l2Penalty;
  regularization.maxSquaredWeightNorm = maxSquaredWeightNorm;
  return *this;
}

Net& Net::setErrorFunction(ErrorFunction errorFunction)
{
  architecture << "error_function " << (int) errorFunction << " ";
  this->errorFunction = errorFunction;
  return *this;
}

void Net::finishedIteration()
{
  bool dropout = this->dropout;
  this->dropout = false;
  if(trainSet)
    trainSet->finishIteration(*this);
  if(validSet)
    validSet->finishIteration(*this);
  this->dropout = dropout;
}

Eigen::VectorXd Net::operator()(const Eigen::VectorXd& x)
{
  tempInput = x.transpose();
  forwardPropagate(0);
  return tempOutput.transpose();
}

Eigen::MatrixXd Net::operator()(const Eigen::MatrixXd& x)
{
  tempInput = x;
  forwardPropagate(0);
  return tempOutput;
}

std::vector<std::vector<double>> Net::operator()(const std::vector<std::vector<double>>& x)
{
    tempInput.resize(x.size(), Eigen::NoChange);
    for(std::size_t r = 0; r < x.size(); r++)
    {
        for(std::size_t c = 0; c < x[r].size(); c++)
        {
            tempInput(r,c) = x[r][c];
        }
    }

    forwardPropagate(0);
    std::size_t dimOut = tempOutput.cols();
    std::vector<std::vector<double>> res(x.size());
    for(std::size_t r = 0; r < res.size(); r++)
    {
        res[r].reserve(dimOut);
        for(std::size_t c = 0; c < (std::size_t)dimOut; c++)
        {
            res[r].push_back(tempOutput(r,c));
        }
    }
    return res;
}

unsigned int Net::dimension()
{
  return P;
}

unsigned int Net::dimension() const
{
  return P;
}

unsigned int Net::examples()
{
  return N;
}

const Eigen::VectorXd& Net::currentParameters() const
{
  return parameterVector;
}

const Eigen::VectorXd& Net::currentParameters()
{
  return parameterVector;
}

void Net::setParameters(const Eigen::VectorXd& newParameters)
{
  if(newParameters.hasNaN())
  {
    OPENANN_ERROR << "Error - parameters contain NaN's! Ignoring...\n";
    return;
  }

  if(newParameters.array().isInf().any())
  {
    OPENANN_ERROR << "Error - parameters contain Infs's! Ignoring...\n";
    return;
  }

  if(newParameters.size() != this->parameters.size())
  {
    OPENANN_ERROR << "Error - new parameters have size" << newParameters.size() << " but old parameters have size " << this->parameters.size() << ". Ignoring..." << std::endl;
    return;
  }

  parameterVector = newParameters;
  for(int p = 0; p < P; p++)
    *(this->parameters[p]) = newParameters(p);
  for(std::vector<Layer*>::iterator layer = layers.begin();
      layer != layers.end(); ++layer)
    (**layer).updatedParameters();
}

bool Net::providesInitialization()
{
  return true;
}

void Net::initialize()
{
  OPENANN_CHECK(initialized);
  for(std::vector<Layer*>::iterator layer = layers.begin();
      layer != layers.end(); ++layer)
    (**layer).initializeParameters();
  for(int p = 0; p < P; p++)
    parameterVector(p) = *parameters[p];
}

double Net::error(unsigned int n)
{
  tempInput = trainSet->getInstance(n).transpose();
  double regularizationError = 0;
  forwardPropagate(&regularizationError);
  if(errorFunction == CE)
    return crossEntropy(tempOutput, trainSet->getTarget(n).transpose()) +
        regularizationError;
  else
    return meanSquaredError(tempOutput - trainSet->getTarget(n).transpose()) +
        regularizationError;
}

double Net::error()
{
  double e = 0.0;
  for(int n = 0; n < N; n++)
    e += error(n) / (double) N;
  return e;
}

bool Net::providesGradient()
{
  return true;
}

Eigen::VectorXd Net::gradient(unsigned int n)
{
  std::vector<int> indices;
  indices.push_back(n);
  double error;
  errorGradient(indices.begin(), indices.end(), error, tempGradient);
  return tempGradient;
}

Eigen::VectorXd Net::gradient()
{
  std::vector<int> indices;
  indices.reserve(N);
  for(int n = 0; n < N; n++)
    indices.push_back(n);
  double error;
  errorGradient(indices.begin(), indices.end(), error, tempGradient);
  return tempGradient;
}

void Net::errorGradient(int n, double& value, Eigen::VectorXd& grad)
{
  std::vector<int> indices;
  indices.push_back(n);
  errorGradient(indices.begin(), indices.end(), value, grad);
}

void Net::errorGradient(double& value, Eigen::VectorXd& grad)
{
  std::vector<int> indices;
  for(int n = 0; n < N; n++)
    indices.push_back(n);
  errorGradient(indices.begin(), indices.end(), value, grad);
}

void Net::errorGradient(std::vector<int>::const_iterator startN,
                        std::vector<int>::const_iterator endN,
                        double& value, Eigen::VectorXd& grad)
{
  const int N = endN - startN;

  if(N <= 0)
  {
    OPENANN_ERROR << "Number of samples is " << N << ". Aborting...\n";
    grad.setZero();
    value = 0;
    return;
  }


  tempInput.conservativeResize(N, trainSet->inputs());
  Eigen::MatrixXd T(N, trainSet->outputs());
  int n = 0;
  for(std::vector<int>::const_iterator it = startN; it != endN; ++it, ++n)
  {
    tempInput.row(n) = trainSet->getInstance(*it);
    T.row(n) = trainSet->getTarget(*it);
  }

  value = 0;
  forwardPropagate(&value);
  tempError = tempOutput - T;  //TODO: Does this mean that we always use the quadratic error? We should see here the derivation of the cost function to the output of the individual neurons
  value += errorFunction == CE ? crossEntropy(tempOutput, T) :
      meanSquaredError(tempError);
  backpropagate();

  for(int p = 0; p < P; p++)
    grad(p) = *derivatives[p];
  grad /= N;
}

void Net::initializeNetwork()
{
  if(initialized)
    throw OpenANNException("Tried to initialize network again.");

  P = parameters.size();
  tempInput.resize(1, infos[0].outputs());
  tempOutput.resize(1, infos.back().outputs());
  tempError.resize(1, infos.back().outputs());
  tempGradient.resize(P);
  parameterVector.resize(P);
  for(int p = 0; p < P; p++)
    parameterVector(p) = *parameters[p];
  initialized = true;
}

void Net::forwardPropagate(double* error)
{

  Eigen::MatrixXd* y = &tempInput;
  int cntLayer = 0;
  for(std::vector<Layer*>::iterator layer = layers.begin();layer != layers.end(); ++layer)
  {
    (**layer).forwardPropagate(y, y, dropout, error);
    cntLayer++;
  }

  tempOutput = *y;
  OPENANN_CHECK_EQUALS(y->cols(), infos.back().outputs());
  if(errorFunction == CE)
    OpenANN::softmax(tempOutput);
}

void Net::backpropagate()
{
  Eigen::MatrixXd* e = &tempError;

  int l = L;
  for(std::vector<Layer*>::reverse_iterator layer = layers.rbegin();
      layer != layers.rend(); ++layer, --l)
  {
    // Backprop of dE/dX is not required in input layer and first hidden layer
    bool backpropToPrevious = true;
    if(!this->backpropToAll)
        backpropToPrevious = l > 2;

    (**layer).backpropagate(e, e, backpropToPrevious);

  }

  tempError = *e;
}

Eigen::MatrixXd Net::getLayerError() const
{
    return tempError;
}

Eigen::VectorXd Net::currentGradients()
{
    Eigen::VectorXd grad(P);

    for(int p = 0; p < P; p++)
        grad(p) = *derivatives[p];

    return grad;
}

}
