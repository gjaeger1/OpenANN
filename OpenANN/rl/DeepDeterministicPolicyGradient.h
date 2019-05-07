#pragma once
#include <OpenANN/io/DirectStorageDataSet.h>
#include <OpenANN/rl/Environment.h>
#include <OpenANN/OpenANN>

namespace OpenANN
{
  namespace rl
  {
    class ReplayBuffer
    {
      protected:
        std::size_t insertIndex;
        std::size_t bufferSize;
        std::size_t maxBufferSize;

        Eigen::MatrixXd states;
        Eigen::MatrixXd actions;
        Eigen::VectorXd rewards;
        Eigen::MatrixXd newStates;

        int checkIndex(const int& index);

      public:
        ReplayBuffer(const std::size_t& maxBufferSize, const std::size_t& stateDim, const std::size_t& actionDim);

        virtual void addSample(const Eigen::VectorXd& state, const Eigen::VectorXd& action, const double& reward, const Eigen::VectorXd& newState);

        virtual int samples() const {return this->bufferSize;};

        virtual Eigen::VectorXd getState(int n);
        virtual Eigen::VectorXd getAction(int n);
        virtual double getReward(int n);
        virtual Eigen::VectorXd getNewState(int n);

    };

    struct DDPG
    {
      std::size_t bufferSize = 100;
      double stddevNoise = 1.0;
      double tau = 0.001;
      bool initNetworks = true;
      std::size_t maxEpisodes = 1000;
      std::size_t maxTimeStepsInEpisode = 1000;
      int miniBatchSize = 10;
      double learningRate = 0.01;
      std::size_t updateSteps = 50;
      double discountFactor = 0.99;

      void learn(OpenANN::Net& critic, OpenANN::Net& actor, OpenANN::Environment& environment);
    };
  }
}
