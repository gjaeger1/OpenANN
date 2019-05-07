#include <iostream>
#include <OpenANN/rl/DeepDeterministicPolicyGradient.h>
#include "SinglePoleBalancing.h"
#include <OpenANN/OpenANN>

int main(int argc, char** argv)
{
    SinglePoleBalancing spb(true);
    OpenANN::Net actor;
    actor.inputLayer(spb.stateSpaceDimension())
    .rbfLayer(101)
    .outputLayer(1, OpenANN::LINEAR);

    OpenANN::Net critic;
    critic.inputLayer(spb.actionSpaceDimension()+spb.stateSpaceDimension())
    .rbfLayer(101)
    .outputLayer(1, OpenANN::LINEAR);

    OpenANN::rl::DDPG ddpb;

    ddpb.learn(critic, actor, spb);

    return 0;
}
