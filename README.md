# Pytorch Snake AI

# A Snake AI Agent trained using Reinforcement Learning

Simple 1 hidden linear layer network used.

Next iteration: a `ConvNet` taking the frame image as input

This iteration: **Not superhuman** but *pretty* good. Need a convnet for that.

Few trained models in models/ directory to play around with.

Play snake yourself ( Vim Controls (sorry!) ):

```
python3 -m game.game_human
```

Test trained model:

```
./test.py
```

or test the average score over N games with:

```
./test.py 100
```

Play around with training hyper-parameters and re-train a model:

```
./train.py
```
