#ifndef OPENANN_OPTIMIZATION_AdamSGD_H_
#define OPENANN_OPTIMIZATION_AdamSGD_H_

#include <OpenANN/optimization/Optimizer.h>
#include <OpenANN/optimization/StoppingCriteria.h>
#include <OpenANN/util/Random.h>
#include <Eigen/Core>
#include <vector>
#include <list>

namespace OpenANN
{

/**
 * @class AdamSGD
 *
 * Mini-batch stochastic gradient descent with Adam update rule.
 *
 * This implementation of gradient descent has some modifications:
 *
 * - it is stochastic, we update the weights with a randomly chosen subset of
 *   the training set to escape local minima and thus increase the
 *   generalization capabilities
 * - we use a momentum to smooth the search direction
 * - each weight has an adaptive learning rate
 * - the learning rate and momentum are adpated during the optimization using the Adam rule.
 */
class AdamSGD : public Optimizer
{
  //! Stopping criteria
  StoppingCriteria stop;
  //! Optimizable problem
  Optimizable* opt; // do not delete
  //! Basic learning rate that will be adjusted over time
  double learningRate;
  //! Beta 1 - Parameter of Adam update-rule
  double beta1;
  //! Beta 2 - Parameter of Adam update-rule
  double beta2;
  //! Size of a single batch, that is, number of training samples used in each step of the training.
  int batchSize;

  int iteration;
  RandomNumberGenerator rng;
  int P, N, batches;
  Eigen::VectorXd gradient, currentGradient, parameters, vt, mt;
  double accumulatedError;
  std::vector<int> randomIndices;

  static constexpr double epsilonAdam = 0.000000010;
public:
  /**
   * Create mini-batch stochastic gradient descent optimizer.
   *
   * @param learningRate learning rate (usually called alpha); range: (0, 1]
   * @param beta1 Beta1 Parameter of Adam update rule
   * @param beta2 Beta2 Parameter of Adam update rule
   * @param batchSize size of the mini-batches; range: [1, N], where N is the
   *                  size of the training set
   */
  AdamSGD(double learningRate = 0.01, double beta1 = 0.9, double beta2 = 0.999, int batchSize = 10);
  ~AdamSGD();
  virtual void setOptimizable(Optimizable& opt);
  virtual void setStopCriteria(const StoppingCriteria& stop);
  virtual void optimize();
  virtual bool step();
  virtual Eigen::VectorXd result();
  virtual std::string name();
private:
  void initialize();
};

} // namespace OpenANN

#endif // OPENANN_OPTIMIZATION_AdamSGD_H_
