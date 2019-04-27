#ifndef OPENANN_TEST_RBF_TEST_CASE_H_
#define OPENANN_TEST_RBF_TEST_CASE_H_

#include <Test/TestCase.h>

class RBFTestCase : public TestCase
{
  virtual void run();
  void forward();
  void backprop();
  void inputGradient();
  void parallelForward();
  void regularization();
};

#endif // OPENANN_TEST_RBF_TEST_CASE_H_
