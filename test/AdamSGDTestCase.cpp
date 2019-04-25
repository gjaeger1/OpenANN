#include "AdamSGDTestCase.h"
#include "optimization/Quadratic.h"
#include <OpenANN/optimization/AdamSGD.h>
#include <OpenANN/io/Logger.h>

void AdamSGDTestCase::run()
{
  RUN(AdamSGDTestCase, quadratic);
  RUN(AdamSGDTestCase, restart);
}

void AdamSGDTestCase::quadratic()
{
  OpenANN::AdamSGD adam;
  Quadratic<10> q;
  q.setParameters(Eigen::VectorXd::Ones(10));
  OpenANN::StoppingCriteria s;
  s.maximalIterations = 1000;
  s.minimalSearchSpaceStep = 1e-10;
  adam.setOptimizable(q);
  adam.setStopCriteria(s);
  adam.optimize();
  Eigen::VectorXd optimum = adam.result();
  ASSERT(q.error() < 0.001);
}

void AdamSGDTestCase::restart()
{
  OpenANN::AdamSGD adam;
  Quadratic<10> q;
  q.setParameters(Eigen::VectorXd::Ones(10));
  OpenANN::StoppingCriteria s;
  s.maximalIterations = 1000;
  s.minimalSearchSpaceStep = 1e-10;
  adam.setOptimizable(q);
  adam.setStopCriteria(s);
  adam.optimize();
  Eigen::VectorXd optimum = adam.result();
  ASSERT(q.error() < 0.001);

  // Restart
  q.setParameters(Eigen::VectorXd::Ones(10));
  ASSERT(q.error() == 10.0);
  adam.optimize();
  optimum = adam.result();
  ASSERT(q.error() < 0.001);
}
