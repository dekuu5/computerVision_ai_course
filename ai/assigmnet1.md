# assigment1 


## 10 types of optimizers used in Machine Learning

1. Stochastic Gradient Descent (SGD): Updates model parameters using the gradient calculated from a single training sample at a time.

2. Batch Gradient Descent: Updates parameters using the gradient calculated from the entire training dataset in one go.

3. Mini-Batch Gradient Descent: A compromise that updates parameters using the gradient from a small, random batch of training samples.

4. Momentum: Helps accelerate SGD by adding a fraction of the previous update direction to the current one, dampening oscillations.

5. Nesterov Accelerated Gradient (NAG): A "smarter" version of Momentum that "looks ahead" by calculating the gradient at its projected future position.

6. Adagrad (Adaptive Gradient): Adapts the learning rate for each parameter, using smaller updates for frequent parameters and larger updates for infrequent ones.

7. RMSprop (Root Mean Square Propagation): Fixes Adagrad's diminishing learning rate by using a moving average of squared gradients instead of accumulating them.

8. Adadelta: An extension of Adagrad that also addresses the diminishing learning rate, but without needing a base learning rate set manually.

9. Adam (Adaptive Moment Estimation): Combines the ideas of Momentum (storing a moving average of past gradients) and RMSprop (storing a moving average of past squared gradients).

10. AdamW: A variation of Adam that improves regularization by decoupling the weight decay calculation from the gradient update step.