# Deep Recurrent Q-Network
This is a demo of DQRN, in the Mazeworld.

In the maze: Mazeworld3, there is a BONUS at position (3, 4), if you get (3, 4) for the first time, and if you choose to act "UP", you will get a bonus of 100.

For a DQN agent, it can only choose to go "RIGHT", for the policy has to be DETERMINISTIC. For a DRQN agent, it can learn act "UP" at the first time reaching (3, 4), and then go "RIGHT".

The codes for this project is based on DQN and DRQN at https://github.com/metalbubble/DeepRL-Tutorials.
The idea of DRQN is from: https://arxiv.org/abs/1507.06527
