#define OPENANN_LOG_NAMESPACE "AdamSGD"

#include <OpenANN/optimization/AdamSGD.h>
#include <OpenANN/optimization/Optimizable.h>
#include <OpenANN/optimization/StoppingCriteria.h>
#include <OpenANN/optimization/StoppingInterrupt.h>
#include <OpenANN/util/AssertionMacros.h>
#include <OpenANN/util/OpenANNException.h>
#include <OpenANN/util/EigenWrapper.h>
#include <OpenANN/io/Logger.h>
#include <Test/Stopwatch.h>
#include <numeric>

namespace OpenANN
{

AdamSGD::AdamSGD(double learningRate, double beta1, double beta2, int batchSize):
    opt(0), learningRate(learningRate), beta1(beta1), beta2(beta2), iteration(-1), batchSize(batchSize),
    P(-1), N(-1), batches(-1),accumulatedError(0.0)
{
  if(learningRate <= 0.0 || learningRate > 1.0)
    throw OpenANNException("Invalid learning rate, should be within (0, 1]");
  if(batchSize < 1)
    throw OpenANNException("Invalid batch size, should be greater than 0");
  if(beta1 <= 0.0 || beta1 > 1.0)
    throw OpenANNException("Invalid value for beta1, should be within (0, 1]");
  if(beta2 <= 0.0 || beta2 > 1.0)
    throw OpenANNException("Invalid value for beta2, should be within (0, 1]");
}

AdamSGD::~AdamSGD()
{
}

void AdamSGD::setOptimizable(Optimizable& opt)
{
  this->opt = &opt;
}

void AdamSGD::setStopCriteria(const StoppingCriteria& stop)
{
  this->stop = stop;
}

void AdamSGD::optimize()
{
  OPENANN_CHECK(opt);
  StoppingInterrupt interrupt;
  double bestError = std::numeric_limits<double>::max()/2.0;
  Eigen::VectorXd bestParameters = opt->currentParameters();
  accumulatedError = 0.0;
  iteration = -1;
  while(step() && 2.0*bestError > accumulatedError)
  {
    std::stringstream ss;

    ss << "Iteration " << iteration;
    ss << ", error = " << FloatingPointFormatter(accumulatedError /
                                                 (double) batches, 4);

    if(accumulatedError < bestError)
    {
        bestError = accumulatedError;
        bestParameters = parameters;
    }

    OPENANN_DEBUG << ss.str();
  }

  parameters = bestParameters;
  this->result();
}

bool AdamSGD::step()
{
  OPENANN_CHECK(opt);
  if(iteration < 0)
    initialize();
  OPENANN_CHECK(P > 0);
  OPENANN_CHECK(N > 0);
  OPENANN_CHECK(batches > 0);

  accumulatedError = 0.0;
  // generate random indices of training samples to use in this batch
  rng.generateIndices<std::vector<int> >(N, randomIndices, true);
  std::vector<int>::const_iterator startN = randomIndices.begin();
  std::vector<int>::const_iterator endN = randomIndices.begin() + batchSize;
  if(endN > randomIndices.end())
    endN = randomIndices.end();


  for(int b = 0; b < batches; b++)
  {

    double error = 0.0;
    opt->errorGradient(startN, endN, error, gradient);
    accumulatedError += error;
    OPENANN_CHECK_MATRIX_BROKEN(gradient);

    if(gradient.hasNaN())
    {
        OPENANN_ERROR << "NaNs in gradient detected!\n";
        return false;
    }

    if(gradient.array().isInf().any())
    {
        OPENANN_ERROR << "Infs in gradient detected!\n";
        return false;
    }

    // update velocities and momentum
    mt = mt.cwiseProduct(mt*beta1) + gradient*(1.0-beta1);
    vt = vt.cwiseProduct(vt*beta2) + gradient.cwiseProduct(gradient)*(1.0-beta2);

    // correct bias in velocities and momentum
    Eigen::VectorXd mt_cor = mt * (1.0/(1.0 - std::pow(beta1, std::max(1,iteration+b))));
    Eigen::VectorXd vt_cor = vt * (1.0/(1.0 - std::pow(beta2, std::max(1,iteration+b))));
    vt_cor = vt_cor.cwiseSqrt().array() + epsilonAdam;
    vt_cor = vt_cor.unaryExpr([](double v) { return std::isfinite(v)? v : 1.0; });
    mt_cor = mt_cor.unaryExpr([](double v) { return std::isfinite(v)? v : 0.0; });
    Eigen::VectorXd updates = learningRate*vt_cor.cwiseInverse().cwiseProduct(mt_cor);

    OPENANN_CHECK_MATRIX_BROKEN(updates);

    // update parameters
    parameters -= updates;

    OPENANN_CHECK_MATRIX_BROKEN(parameters);

    if(parameters.hasNaN())
    {
        OPENANN_ERROR << "New parameters have NaNs!\n";
        return false;
    }

    if(parameters.array().isInf().any())
    {
        OPENANN_ERROR << "New parameters have Infs!\n";
        return false;
    }

    opt->setParameters(parameters);

    startN += batchSize;
    endN += batchSize;
    if(endN > randomIndices.end())
      endN = randomIndices.end();
  }

  iteration++;

  opt->finishedIteration();

  const bool run = (stop.maximalIterations == // Maximum iterations reached?
                    StoppingCriteria::defaultValue.maximalIterations ||
                    iteration < stop.maximalIterations) &&
                   (stop.minimalSearchSpaceStep == // Gradient too small?
                    StoppingCriteria::defaultValue.minimalSearchSpaceStep ||
                    gradient.norm() >= stop.minimalSearchSpaceStep);
  if(!run)
    iteration = -1;
  return run;
}

Eigen::VectorXd AdamSGD::result()
{
  opt->setParameters(parameters);
  return parameters;
}

std::string AdamSGD::name()
{
  std::stringstream ss;

  ss << "Mini-Batch Stochastic Gradient Descent with Adam update rule";
  ss << "(learning rate = " << learningRate
     << ", beta1 = " << beta1
     << ", beta2 = " << beta2
     << ", batch_size " << batchSize
     << ")";

  return ss.str();
}

void AdamSGD::initialize()
{
  P = opt->dimension();
  N = opt->examples();
  batches = std::max(N / batchSize, 1);
  gradient.resize(P);
  gradient.setZero();
  currentGradient.resize(P);
  vt.resize(P);
  vt.setZero();
  mt.resize(P);
  mt.setZero();
  parameters = opt->currentParameters();
  randomIndices.clear();
  randomIndices.reserve(N);
  rng.generateIndices<std::vector<int> >(N, randomIndices);
  iteration = 0;
}

}
