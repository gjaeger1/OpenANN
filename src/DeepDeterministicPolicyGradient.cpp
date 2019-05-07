#include <OpenANN/rl/DeepDeterministicPolicyGradient.h>
#include <OpenANN/util/OpenANNException.h>
#include <OpenANN/util/AssertionMacros.h>
#include <OpenANN/util/Random.h>

#include <random>
#include <time.h>

namespace OpenANN
{
    namespace rl
    {
        ReplayBuffer::ReplayBuffer(const std::size_t& maxBufferSize, const std::size_t& stateDim, const std::size_t& actionDim) :
            insertIndex(0), bufferSize(0), maxBufferSize(maxBufferSize), states(maxBufferSize, stateDim), actions(maxBufferSize, actionDim), rewards(maxBufferSize), newStates(maxBufferSize, stateDim)
        {}

        int ReplayBuffer::checkIndex(const int& index)
        {
            int n = index;
            if(n > this->bufferSize)
            {
                OPENANN_ERROR << "Asked for " << n << "-th instance. Truncated to " << this->bufferSize-1 << "\n";
                n = this->bufferSize-1;
            }

            if(n < 0)
            {
                OPENANN_ERROR << "Asked for " << n << "-th instance. Truncated to 0\n";
                n = 0;
            }

            if(n >= this->bufferSize)
            {
                OPENANN_ERROR << "n was " << n << " but buffer is only of size " << this->bufferSize << ". Aborting...\n";
                throw OpenANNException("Tried to read not valid instance from DataBuffer");
            }

            return n;
        }

        Eigen::VectorXd ReplayBuffer::getAction(int n)
        {

            n = this->checkIndex(n);
            return this->actions.row(n);
        }

        Eigen::VectorXd ReplayBuffer::getState(int n)
        {
            n = this->checkIndex(n);
            return this->states.row(n);
        }

        Eigen::VectorXd ReplayBuffer::getNewState(int n)
        {
            n = this->checkIndex(n);
            return this->newStates.row(n);
        }

        double ReplayBuffer::getReward(int n)
        {
            n = this->checkIndex(n);
            return this->rewards(n);
        }

        void ReplayBuffer::addSample(const Eigen::VectorXd& state, const Eigen::VectorXd& action, const double& reward, const Eigen::VectorXd& newState)
        {
            this->states.row(this->insertIndex) = state;
            this->actions.row(this->insertIndex) = action;
            this->rewards(this->insertIndex) = reward;
            this->newStates.row(this->insertIndex) = newState;

            this->insertIndex++;

            if(this->bufferSize < this->maxBufferSize)
                this->bufferSize++;

            if(this->insertIndex >= this->maxBufferSize)
                this->insertIndex = 0;
        }

        void DDPG::learn(OpenANN::Net& critic, OpenANN::Net& actor, OpenANN::Environment& environment)
        {
            // randomize networks
            if(this->initNetworks)
            {
                critic.initialize();
                actor.initialize();
            }

            // initialize target networks as copies of actor and critic
            OpenANN::Net criticPrime = critic;
            OpenANN::Net actorPrime = actor;

            // initialize replay buffer
            ReplayBuffer buffer(this->bufferSize, environment.stateSpaceDimension(), environment.actionSpaceDimension());
            std::vector<int> indices;
            indices.reserve(this->miniBatchSize);

            // init random number generator
            RandomNumberGenerator rng;

            for(std::size_t episode = 0; episode < this->maxEpisodes; episode++)
            {
                // init random process here
                std::default_random_engine generator (std::rand());
                std::normal_distribution<double> distribution (0.0,this->stddevNoise);

                // init environment
                environment.restart();
                Environment::State s = environment.getState();
                for(std::size_t t = 0; t < this->maxTimeStepsInEpisode; t++)
                {
                    Environment::Action a = actor(s);

                    // add randomness for exploration
                    for(std::size_t i = 0; i < a.size(); i++)
                        a(i) += distribution(generator);

                    environment.stateTransition(a);

                    Environment::State ns = environment.getState();
                    double reward = environment.reward();

                    buffer.addSample(s,a,reward,ns);

                    bool update = true;
                    if(update)
                    {
                        for(std::size_t u = 0; u < this->updateSteps; u++)
                        {
                            // draw mini-batch from replay buffer
                            rng.generateIndices<std::vector<int> >(buffer.samples(), indices, false);
                            std::vector<int>::const_iterator startN = indices.begin();
                            std::vector<int>::const_iterator endN = indices.begin() + this->miniBatchSize;
                            if(endN > indices.end())
                              endN = indices.end();

                            Eigen::MatrixXd sampledStates(this->miniBatchSize, environment.stateSpaceDimension());
                            Eigen::MatrixXd sampledActions(this->miniBatchSize, environment.actionSpaceDimension());
                            Eigen::MatrixXd sampledNewStates(this->miniBatchSize, environment.stateSpaceDimension());
                            Eigen::VectorXd sampledRewards(this->miniBatchSize);
                            std::size_t cnt = 0;
                            while(startN != endN)
                            {
                                sampledStates.row(cnt) = buffer.getState(*startN);
                                sampledActions.row(cnt) = buffer.getAction(*startN);
                                sampledNewStates.row(cnt) = buffer.getNewState(*startN);
                                sampledRewards(cnt) = buffer.getReward(*startN);
                                cnt++;
                                startN++;
                            }

                            indices.clear();

                            // generate targets
                            Eigen::MatrixXd inputsCritic(this->miniBatchSize,environment.actionSpaceDimension()+environment.stateSpaceDimension());
                            inputsCritic.leftCols(environment.actionSpaceDimension()) = actorPrime(sampledNewStates);
                            inputsCritic.rightCols(environment.stateSpaceDimension()) = sampledNewStates;
                            Eigen::MatrixXd yi = sampledRewards + this->discountFactor * criticPrime(inputsCritic);


                            // update critic using vanilla SGD
                            Eigen::VectorXd gradCritic(critic.dimension());
                            critic.trainingSet(inputsCritic, yi);
                            double v = 0;
                            critic.errorGradient(v, gradCritic);
                            Eigen::MatrixXd nparams = critic.currentParameters() - this->learningRate*gradCritic;
                            critic.setParameters(nparams);

                            // update actor
                            Eigen::MatrixXd tempError = critic.getLayerError().rightCols(environment.actionSpaceDimension());
                            Eigen::MatrixXd* e = &tempError;
                            for(int l = actor.numberOflayers(); l > 0; l--)
                            {
                                actor.getLayer(l-1).backpropagate(e,e, true);
                            }

                            Eigen::MatrixXd cgrads = (this->learningRate/(double)this->miniBatchSize)*actor.currentGradients();
                            Eigen::MatrixXd anparams = actor.currentParameters();
                            anparams = anparams - cgrads;
                            actor.setParameters(anparams);



                            // update target networks
                            Eigen::VectorXd cparams = this->tau*actorPrime.currentParameters() + (1.0 - this->tau)*actor.currentParameters();
                            actorPrime.setParameters(cparams);

                            cparams = this->tau*criticPrime.currentParameters() + (1.0 - this->tau)*critic.currentParameters();
                            criticPrime.setParameters(cparams);
                        }
                    }
                }
            }

        }
    }
}
