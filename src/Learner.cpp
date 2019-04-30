#include <OpenANN/Learner.h>
#include <OpenANN/io/DirectStorageDataSet.h>

namespace OpenANN
{

Learner::Learner()
  : trainSet(0), validSet(0), deleteTrainSet(false), deleteValidSet(false),
    N(0)
{
}

Learner::~Learner()
{
  this->clearDatasets();
}

void Learner::clearDatasets()
{
  if(deleteTrainSet && trainSet)
    delete trainSet;
  if(deleteValidSet && validSet)
    delete validSet;
}

/**
* @brief Copy Constructor
*/
Learner::Learner(const Learner& other)
{
    this->trainSet = other.trainSet;
    this->validSet = other.validSet;
    this->N = other.N;

    this->deleteTrainSet = false;
    this->deleteValidSet = false;
}

/**
* @brief Move Constructor
*/
Learner::Learner(Learner&& other): trainSet(0), validSet(0), deleteTrainSet(false), deleteValidSet(false),
    N(0)
{
    this->clearDatasets();

    std::swap(this->trainSet, other.trainSet);
    std::swap(this->validSet, other.validSet);
    std::swap(this->N,other.N);
    std::swap(this->deleteTrainSet, other.deleteTrainSet);
    std::swap(this->deleteValidSet, other.deleteValidSet);
}

/**
* @brief Copy Assignment Operator
*/
Learner& Learner::operator = (const Learner& other)
{
    this->trainSet = other.trainSet;
    this->validSet = other.validSet;
    this->N = other.N;

    this->deleteTrainSet = false;
    this->deleteValidSet = false;

    return *this;
}

/**
* @brief Move Assignment Operator
*/
Learner& Learner::operator = (Learner&& other)
{
    this->clearDatasets();

    std::swap(this->trainSet, other.trainSet);
    std::swap(this->validSet, other.validSet);
    std::swap(this->N,other.N);
    std::swap(this->deleteTrainSet, other.deleteTrainSet);
    std::swap(this->deleteValidSet, other.deleteValidSet);

    return *this;
}

Learner& Learner::trainingSet(Eigen::MatrixXd& input,
                              Eigen::MatrixXd& output)
{
  if(deleteTrainSet && trainSet)
    delete trainSet;
  trainSet = new DirectStorageDataSet(&input, &output);
  deleteTrainSet = true;
  N = trainSet->samples();
  return *this;
}

Learner& Learner::trainingSet(DataSet& trainingSet)
{
  if(deleteTrainSet && trainSet)
    delete trainSet;
  trainSet = &trainingSet;
  deleteTrainSet = false;
  N = trainSet->samples();
  return *this;
}

Learner& Learner::removeTrainingSet()
{
  if(deleteTrainSet && trainSet)
    delete trainSet;
  deleteTrainSet = false;
  trainSet = 0;
  N = 0;
  return *this;
}

Learner& Learner::validationSet(Eigen::MatrixXd& input,
                                Eigen::MatrixXd& output)
{
  if(deleteValidSet && validSet)
    delete validSet;
  validSet = new DirectStorageDataSet(&input, &output);
  deleteValidSet = true;
  return *this;
}

Learner& Learner::validationSet(DataSet& validationSet)
{
  if(deleteValidSet && validSet)
    delete validSet;
  validSet = &validationSet;
  deleteValidSet = false;
  return *this;
}

Learner& Learner::removeValidationSet()
{
  if(deleteValidSet && validSet)
    delete validSet;
  deleteValidSet = false;
  validSet = 0;
  return *this;
}

}
