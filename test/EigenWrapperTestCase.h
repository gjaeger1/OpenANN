#ifndef OPENANN_TEST_EIGEN_WRAPPER_TEST_CASE_H_
#define OPENANN_TEST_EIGEN_WRAPPER_TEST_CASE_H_

#include "Test/TestCase.h"

class EigenWrapperTestCase : public TestCase
{
  virtual void run();
  void stdVectorConversion();
};

#endif // OPENANN_TEST_EIGEN_WRAPPER_TEST_CASE_H_
