#include "RBFTestCase.h"
#include "LayerAdapter.h"
#include "FiniteDifferences.h"
#include <OpenANN/layers/RBF.h>

void RBFTestCase::run()
{
  RUN(RBFTestCase, forward);
  RUN(RBFTestCase, backprop);
  RUN(RBFTestCase, inputGradient);
  RUN(RBFTestCase, parallelForward);
  RUN(RBFTestCase, regularization);
}

void RBFTestCase::forward()
{
  OpenANN::OutputInfo info;
  info.dimensions.push_back(3);
  OpenANN::RBF layer(info, 2, 0.05,
                                OpenANN::Regularization());

  std::vector<double*> pp;
  std::vector<double*> pdp;
  OpenANN::OutputInfo info2 = layer.initialize(pp, pdp);
  ASSERT_EQUALS(info2.dimensions.size(), 1);
  ASSERT_EQUALS(info2.outputs(), 2);

  for(std::vector<double*>::iterator it = pp.begin(); it != pp.end(); ++it)
    **it = 1.0;

  Eigen::MatrixXd x(1, 3);
  x << 0.5, 1.0, 2.0;
  Eigen::MatrixXd xgrad(1, 3);
  xgrad << 0.25, 0, 1.0;
  Eigen::MatrixXd e(1, 2);
  e << 1.0, 2.0;

  Eigen::MatrixXd* y = 0;
  layer.forwardPropagate(&x, y, false);
  ASSERT(y != 0);
  double a = std::exp(-1.25);
  ASSERT_EQUALS_DELTA((*y)(0, 0), a, 1e-10);
  ASSERT_EQUALS_DELTA((*y)(0, 1), a, 1e-10);

  Eigen::MatrixXd* e2;
  layer.backpropagate(&e, e2, true);
  Eigen::VectorXd Wd(8);
  int i = 0;
  for(std::vector<double*>::iterator it = pdp.begin(); it != pdp.end(); ++it)
    Wd(i++) = **it;

  // j == 0
  ASSERT_EQUALS_DELTA(Wd(0), 2.0*(1.0-0.5)*1.0*1.0*-a, 1e-7);
  ASSERT_EQUALS_DELTA(Wd(1), 2.0*(1.0-1.0)*1.0*1.0*-a, 1e-7);
  ASSERT_EQUALS_DELTA(Wd(2), 2.0*(1.0-2.0)*1.0*1.0*-a, 1e-7);
  ASSERT_EQUALS_DELTA(Wd(3), 1.25*1.0*-a, 1e-7); // bias

  // j == 1
  ASSERT_EQUALS_DELTA(Wd(4), 2.0*(1.0-0.5)*1.0*2.0*-a, 1e-7);
  ASSERT_EQUALS_DELTA(Wd(5), 2.0*(1.0-1.0)*1.0*2.0*-a, 1e-7);
  ASSERT_EQUALS_DELTA(Wd(6), 2.0*(1.0-2.0)*1.0*2.0*-a, 1e-7);
  ASSERT_EQUALS_DELTA(Wd(7), 1.25*2.0*-a, 1e-7); // bias
  ASSERT(e2 != 0);
}

void RBFTestCase::backprop()
{
  OpenANN::OutputInfo info;
  info.dimensions.push_back(3);
  OpenANN::RBF layer(info, 2, 0.05,
                                OpenANN::Regularization());
  LayerAdapter opt(layer, info);

  Eigen::MatrixXd X = Eigen::MatrixXd::Random(2, 3);
  Eigen::MatrixXd Y = Eigen::MatrixXd::Random(2, 2);
  std::vector<int> indices;
  indices.push_back(0);
  indices.push_back(1);
  opt.trainingSet(X, Y);
  Eigen::VectorXd gradient = opt.gradient(indices.begin(), indices.end());
  Eigen::VectorXd estimatedGradient = OpenANN::FiniteDifferences::
      parameterGradient(indices.begin(), indices.end(), opt);
  for(int i = 0; i < gradient.rows(); i++)
    ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), 1e-7);
}

void RBFTestCase::inputGradient()
{
  OpenANN::OutputInfo info;
  info.dimensions.push_back(3);
  OpenANN::RBF layer(info, 2, 0.05,
                                OpenANN::Regularization());
  LayerAdapter opt(layer, info);

  Eigen::MatrixXd X = Eigen::MatrixXd::Random(2, 3);
  Eigen::MatrixXd Y = Eigen::MatrixXd::Random(2, 2);
  opt.trainingSet(X, Y);
  Eigen::MatrixXd gradient = opt.inputGradient();
  ASSERT_EQUALS(gradient.rows(), 2);
  Eigen::MatrixXd estimatedGradient = OpenANN::FiniteDifferences::
      inputGradient(X, Y, opt);
  ASSERT_EQUALS(estimatedGradient.rows(), 2);
  for(int j = 0; j < gradient.rows(); j++)
    for(int i = 0; i < gradient.cols(); i++)
      ASSERT_EQUALS_DELTA(gradient(j, i), estimatedGradient(j, i), 1e-10);
}

void RBFTestCase::parallelForward()
{
  OpenANN::OutputInfo info;
  info.dimensions.push_back(3);
  OpenANN::RBF layer(info, 2, 0.05,
                                OpenANN::Regularization());

  std::vector<double*> pp;
  std::vector<double*> pdp;
  OpenANN::OutputInfo info2 = layer.initialize(pp, pdp);
  ASSERT_EQUALS(info2.dimensions.size(), 1);
  ASSERT_EQUALS(info2.outputs(), 2);

  for(std::vector<double*>::iterator it = pp.begin(); it != pp.end(); ++it)
    **it = 1.0;
  Eigen::MatrixXd x(2, 3);
  x << 0.5, 1.0, 2.0,
       0.5, 1.0, 2.0;
  Eigen::MatrixXd xgrad(1, 3);
  xgrad << 1.0-0.5, 1.0-1.0, 1.0-2.0;
  Eigen::MatrixXd e(2, 2);
  e << 1.0, 2.0,
       1.0, 2.0;

  Eigen::MatrixXd* y = 0;
  layer.forwardPropagate(&x, y, false);
  ASSERT(y != 0);
  double a = std::exp(-1.25);
  ASSERT_EQUALS_DELTA((*y)(0, 0), a, 1e-10);
  ASSERT_EQUALS_DELTA((*y)(0, 1), a, 1e-10);
  ASSERT_EQUALS_DELTA((*y)(1, 0), a, 1e-10);
  ASSERT_EQUALS_DELTA((*y)(1, 1), a, 1e-10);

  Eigen::MatrixXd* e2;
  layer.backpropagate(&e, e2, true);
  Eigen::VectorXd Wd(8);
  int i = 0;
  for(std::vector<double*>::iterator it = pdp.begin(); it != pdp.end(); ++it)
    Wd(i++) = **it;

  // j == 0
  ASSERT_EQUALS_DELTA(Wd(0), 2.0*2.0*(1.0-0.5)*1.0*1.0*-a, 1e-7);
  ASSERT_EQUALS_DELTA(Wd(1), 2.0*2.0*(1.0-1.0)*1.0*1.0*-a, 1e-7);
  ASSERT_EQUALS_DELTA(Wd(2), 2.0*2.0*(1.0-2.0)*1.0*1.0*-a, 1e-7);
  ASSERT_EQUALS_DELTA(Wd(3), 2.0*1.25*1.0*-a, 1e-7); // bias

  // j == 1
  ASSERT_EQUALS_DELTA(Wd(4), 2.0*2.0*(1.0-0.5)*1.0*2.0*-a, 1e-7);
  ASSERT_EQUALS_DELTA(Wd(5), 2.0*2.0*(1.0-1.0)*1.0*2.0*-a, 1e-7);
  ASSERT_EQUALS_DELTA(Wd(6), 2.0*2.0*(1.0-2.0)*1.0*2.0*-a, 1e-7);
  ASSERT_EQUALS_DELTA(Wd(7), 2.0*1.25*2.0*-a, 1e-7); // bias
  ASSERT(e2 != 0);
}

void RBFTestCase::regularization()
{
  OpenANN::OutputInfo info;
  info.dimensions.push_back(3);
  OpenANN::RBF layer(info, 2, 0.05,
                                OpenANN::Regularization(0.1, 0.1));
  LayerAdapter opt(layer, info);

  Eigen::MatrixXd X = Eigen::MatrixXd::Random(1, 3);
  Eigen::MatrixXd Y = Eigen::MatrixXd::Random(1, 2);
  opt.trainingSet(X, Y);
  Eigen::VectorXd gradient = opt.gradient(0);
  Eigen::VectorXd estimatedGradient = OpenANN::FiniteDifferences::parameterGradient(0, opt);
  for(int i = 0; i < gradient.rows(); i++)
    ASSERT_EQUALS_DELTA(gradient(i), estimatedGradient(i), 1e-10);
}
