#include "EigenWrapperTestCase.h"
#include <OpenANN/util/EigenWrapper.h>

void EigenWrapperTestCase::run()
{
  RUN(EigenWrapperTestCase, stdVectorConversion);
}

void EigenWrapperTestCase::stdVectorConversion()
{
    int d1 = 3;
    int d2 = 3;

    std::vector<std::vector<double>> v(d1);
    for(int i = 0; i < d1; i++)
    {
        for(int j = 0; j < d2; j++)
            v[i].push_back(i+j);
    }

    Eigen::MatrixXd conv = OpenANN::util::fromStdVector(v);

    for(int i = 0; i < d1; i++)
    {
        for(int j = 0; j < d2; j++)
            ASSERT_EQUALS(v[i][j], conv(i,j));
    }

    std::vector<std::vector<double>> v2 = OpenANN::util::toStdVector(conv);

    for(int i = 0; i < d1; i++)
    {
        for(int j = 0; j < d2; j++)
            ASSERT_EQUALS(v2[i][j], v[i][j]);
    }


}
