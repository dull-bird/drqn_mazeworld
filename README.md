# Deep Recurrent Q-Network
This is a demo of DQRN, with the env as MazeWorld.

* **MazeWorld** (No _mei zi_ here)
1. Basic maze: 
    * IN: (0, 0), OUT: (2, 5)
    * state: (row, col)
    * reward: -5 for any step before finding OUT, 100 for achieving OUT.
    * action: "UP", "DOWN", "Left", "Right", encoded as 0, 1, 2, 3.

2. Bonus maze:
    * IN: (0, 0), OUT: (2, 5)
    * state: (row, col, bonus_bit). bonus_bit indicates whether the bonus has been gotten. We can only see (row, col). Bonus is in a fixed place (3, 4).
    * reward: -5 for any step before finding OUT, 100 for achieving OUT, 100 for getting bonus.
    * action: "UP", "DOWN", "Left", "Right", encoded as 0, 1, 2, 3.

3. Partial-info bonus maze:
    * IN: (0, 0), OUT: (2, 5)
    * state: (row, col, bonus_bit). bonus_bit indicates whether the bonus has been gotten. We can only see (row). Bonus is in a fixed place (3, 4).
    * reward: -5 for any step before finding OUT, 100 for achieving OUT, 100 for getting bonus.
    * action: "UP", "DOWN", "Left", "Right", encoded as 0, 1, 2, 3.
    
* **Networks**

  **Statement:** The codes for this project are based on DQN and DRQN codes at https://github.com/metalbubble/DeepRL-Tutorials.

1. Deep Q-Network (DQN)
    * This is the standard form DQN. See [Deepmind's DQN Nature paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) for ref.
    
2. Deep Recurrent Q-Network (DRQN)
    * The network structure is proposed by [Matthew Hausknecht & Peter Stone](https://arxiv.org/abs/1507.06527).
    * The network uses a RNN (basic RNN, GRU or LSTM) to "remember" or "forget" history states the agent has been to and the new action is based on the "memory".

3. Deep Recurrent Q-Network with **actions** (DRQNA, named by myself)
    * The network is a revision of DRQN. In optimal control we use the _information set I_k_ to help us make decisions, sometimes only history of _S_k_ is not enough. So I put _a_k_ in the RNN.

* **Results**
1. Basic maze
    * All the networks work.

2. Bonus maze
    * For a DQN agent, it can only choose to go "RIGHT", for the policy has to be DETERMINISTIC.
    * For a DRQN agent, it can learn act "UP" at the first time reaching (3, 4), and then go "RIGHT".
    * For a DRQN agent, the same as DRQN.

3. Partial-info bonus maze
    * For a DQN agent, it can't converge (find OUT).
    * For a DRQN agent, it can't converge either.
    * For a DRQNA agent, it can find the optimal path.
