#ifndef OPENANN_IO_DIRECT_STORAGE_DATA_SET_H_
#define OPENANN_IO_DIRECT_STORAGE_DATA_SET_H_
#if __GNUC__ >= 4
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <OpenANN/io/DataSet.h>
#include <OpenANN/io/Logger.h>

class Stopwatch;

namespace OpenANN
{

class Evaluator;

/**
 * @class DirectStorageDataSet
 *
 * Stores the inputs and outputs of the data set directly in two matrices.
 *
 * The data set can log results during optimization.
 */
class DirectStorageDataSet : public DataSet
{
protected:
  const Eigen::MatrixXd* in;
  const Eigen::MatrixXd* out;
  const int N;
  const int D;
  const int F;
  Eigen::VectorXd temporaryInput;
  Eigen::VectorXd temporaryOutput;
  Evaluator* evaluator; //!< Do not delete the evaluator!

public:

  /**
   * Create an instance of DirectStorageDataSet.
   * @param in contains an instance in each row
   * @param out cointains a target in each row
   * @param evaluator monitors optimization progress
   */
  DirectStorageDataSet(const Eigen::MatrixXd* in, const Eigen::MatrixXd* out = 0,
                       Evaluator* evaluator = 0);
  virtual int samples()  const { return N; }
  virtual int inputs()  const { return D; }
  virtual int outputs()  const { return F; }
  virtual Eigen::VectorXd& getInstance(int i);
  virtual Eigen::VectorXd& getTarget(int i);
  virtual void finishIteration(Learner& learner);
  virtual void saveCSV(const std::string& path) const override;
};

} // namespace OpenANN

#endif // OPENANN_IO_DIRECT_STORAGE_DATA_SET_H_
