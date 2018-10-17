# bipedal-walker
Train a Bipedal Robot to walk using Reinforcement Learning

The reinforcement learning technique used is called **Augmented Random Search**

## Dependencies 
* OpenAI Gym (pip install gym)
* Box2D (pip install box2d)
* OpenGL 
* Optional : PyBullet (pip install pybullet)

## Algorithm : Augmented Random Search (ARS)

##### Define Parameters
* steps = 100 : Number of steps in training loop
* episode_length = 2000 : Max number of steps per episode
* learning_rate = 0.01 : How much we update the weights on each iteration
* num_deltas = 16 : Number of variations of random noise generated on each training step
* num_best_deltas = 16 : Number of deltas used to update the weights
* noise = 0.03 : Random noise strength
* seed = 1 : Random seed used for generating the noise
* record_every = 50 : Record a new video after 50 steps

1. We initialize random weights theta
```python
        self.theta = np.zeros((self.output_size, self.input_size))
```
2. Loop
* Generate num_deltas deltas and evaluate positive and negative
* Run num_deltas episodes with pos and neg variations
* Collect rollouts : {r(+), r(-), delta}
* Calculate a standard deviation of all rewards (sigma_rewards)
* Sort the rollouts by maximum rewards and select the best num_best_delta rollouts
* step = sum((r(+) - r(-))*delta) for each best rollout
* theta += learning_rate/(num_best_delta * sigma_rewards*step)
* Evaluate : Play an episode with the new weights and see how we did
Continue until desired performance is reached

3. Video
The video can be found on :
