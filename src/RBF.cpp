#include <OpenANN/layers/RBF.h>
#include <OpenANN/util/Random.h>

namespace OpenANN
{

RBF::RBF(OutputInfo info, int J, double stdDev, Regularization regularization)
  : I(info.outputs()), J(J), stdDev(stdDev),
    W(J, I), Wd(J, I), b(J), bd(J), x(0),dist(1,J), a(1, J), y(1, J), yd(1, J),
    deltas(1, J), e(1, I), regularization(regularization)
{
    e.setZero();
    Wd.setZero();
    bd.setZero();
    dist.setZero();
    yd.setZero();
    deltas.setZero();
}

OutputInfo RBF::initialize(std::vector<double*>& parameterPointers,
                                      std::vector<double*>& parameterDerivativePointers)
{
  parameterPointers.reserve(parameterPointers.size() + J * (I + 1));
  parameterDerivativePointers.reserve(parameterDerivativePointers.size() + J * (I + 1));
  for(int j = 0; j < J; j++)
  {
    for(int i = 0; i < I; i++)
    {
      parameterPointers.push_back(&W(j, i));
      parameterDerivativePointers.push_back(&Wd(j, i));
    }

    parameterPointers.push_back(&b(j));
    parameterDerivativePointers.push_back(&bd(j));
  }

  initializeParameters();

  OutputInfo info;
  info.dimensions.push_back(J);
  return info;
}

void RBF::initializeParameters()
{
  RandomNumberGenerator rng;
  rng.fillNormalDistribution(W, stdDev);
  rng.fillNormalDistribution(b, stdDev);
}

void RBF::updatedParameters()
{
  if(regularization.maxSquaredWeightNorm > 0.0)
  {
    for(int j = 0; j < J; j++)
    {
      const double squaredNorm = W.row(j).squaredNorm();
      if(squaredNorm > regularization.maxSquaredWeightNorm)
        W.row(j) *= sqrt(regularization.maxSquaredWeightNorm / squaredNorm);
    }
  }
}

void RBF::forwardPropagate(Eigen::MatrixXd* x, Eigen::MatrixXd*& y,
                                      bool dropout, double* error)
{
  const int N = x->rows();
  this->y.conservativeResize(N, Eigen::NoChange);
  this->dist.resize(N, Eigen::NoChange);
  this->x = x;

  // Activate neurons
  a.resize(N, Eigen::NoChange);
  for(std::size_t r = 0; r < (std::size_t)x->rows(); r++)
  {
    this->dist.row(r) = (W.rowwise()- x->row(r)).array().square().rowwise().sum().transpose();
    // a = squared_error * epsilon
    a.row(r) = this->dist.row(r).transpose().array() * b.array();
  }

  // Compute output
  this->gaussianActivationFunction(a, this->y);

  // Add regularization error
  if(error && regularization.l1Penalty > 0.0)
    *error += regularization.l1Penalty * W.array().abs().sum();
  if(error && regularization.l2Penalty > 0.0)
    *error += regularization.l2Penalty * W.array().square().sum() / 2.0;
  y = &(this->y);
}

void RBF::backpropagate(Eigen::MatrixXd* ein,
                                   Eigen::MatrixXd*& eout,
                                   bool backpropToPrevious)
{
  const int N = a.rows();
  yd.conservativeResize(N, Eigen::NoChange);

  // Derive activations
  this->gaussianActivationFunctionDerivative(y, yd);
  deltas = yd.cwiseProduct(*ein);

  // Weight derivatives
  Wd.setZero();
  for(std::size_t r = 0; r < (std::size_t)x->rows(); r++)
  {
        Wd += (((W.rowwise()-x->row(r)).array().colwise()* b.array()).colwise() * deltas.array().row(r).transpose()).matrix()* 2.0;
  }

  // bias derivatives
  bd = deltas.cwiseProduct(this->dist).colwise().sum();

  if(regularization.l1Penalty > 0.0)
    Wd.array() += regularization.l1Penalty * W.array() / W.array().abs();
  if(regularization.l2Penalty > 0.0)
    Wd += regularization.l2Penalty * W;

  // Prepare error signals for previous layer
  if(backpropToPrevious)
    this->backpropDeltaFirstPart(*x, deltas, e); // gradient of input-function derived to inputs -> W only for weighted sum

  eout = &e;
}

Eigen::MatrixXd& RBF::getOutput()
{
  return y;
}

Eigen::VectorXd RBF::getParameters()
{
  Eigen::VectorXd p(J*(I+1));
  int idx = 0;
  for(int j = 0; j < J; j++)
    for(int i = 0; i < I; i++)
      p(idx++) = W(j, i);
  for(int j = 0; j < J; j++)
     p(idx++) = b(j);
  return p;
}

void RBF::backpropDeltaFirstPart(const Eigen::MatrixXd& in, const Eigen::MatrixXd& deltas,  Eigen::MatrixXd& ndeltas) const
{
  // loop over samples
  ndeltas.resize(in.rows(), Eigen::NoChange);
  ndeltas.setZero();
  for(std::size_t n = 0; n < in.rows(); n++)
  {
    // a single delta
    for(std::size_t j = 0; j < in.cols(); j++) // j-th neuron in l layer -> layer before this current layer
    {
        for(std::size_t k = 0; k < deltas.cols(); k++) // k-th neuron in l+1 layer -> the current layer
        {
            // actual gradient of euclidean distance function derived to its inputs aj
            auto grad = -2.0*(W(k,j) - in(n,j))*b(k);
            ndeltas(n,j) += deltas(n,k) * grad;
        }
    }
  }
}

}