# Reinforcement-Learning-Q-Learning-Examples

Q learning is off polity learning statergy which used for Reinforcemnt Learning Tasks heavily. The other spec is that Q learning is do its update online. which means you don't need wait until finish the episode to values like monte-carlo.Here you can see the update equation of Q learning.

![alt text](https://i.stack.imgur.com/OMzXf.png)

In Reinforcement Learning you just need to design both problem and solution while you learning. because the solution which means the agent get data from the environment, environment is our problem. Consider the following example.


![alt text](https://www.kdnuggets.com/images/reinforcement-learning-fig3-pacman.gif)

This is very well known game called PacMan. in here the the white dots are the reward for PacMan agent and which need to collect all the rewards before caught by enimies. if enemy caught PacMan which will need to deliver an negative reward to configure the state action table.

So this kind of environment we cannot build easily. So I just used already build environments which available in Open AI GYM.There are Tons of COntrolling environments like Robotics, Atari Games and Box2D controls.
