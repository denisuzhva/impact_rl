# ImpactRL: a reinforcement learning framework for solid body topology optimization

RL models for efficient solid body topology in impact simulations.

Please, run `MainLauncher.py CONFIG1 CONFIG2 ...` to launch training, where `CONFIG1` is a configuration, set up in `./cfg`.
For example,

``
  python MainLauncher.py chess_med_1
``

As of now, two toy example configs are awailable: `chess_small_1` and `chess_med_1`.
These configs task RL agents to learn chessboard patterns.

Unfortunately, finite element method simulations are private, therefore FEM-related configs won't work.
Please, contact me <denis.uzhva@yahoo.com> if you are interested in the finite element part of the project.
