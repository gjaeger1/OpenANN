#ifndef OPENANN_TEST_PCA_TEST_CASE_H_
#define OPENANN_TEST_PCA_TEST_CASE_H_

#include <Test/TestCase.h>

class PCATestCase : public TestCase
{
  virtual void run();
  void decorrelation();
  void dimensionalityReduction();
};

#endif // OPENANN_TEST_PCA_TEST_CASE_H_
