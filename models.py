import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.get_weights(), x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if nn.as_scalar(self.run(x)) < 0:
            return -1
        else:
            return 1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 1 #how many datapoints we want to consider for every update step we take
        w = self.get_weights()
        mistake = False
        for x, y in dataset.iterate_once(batch_size):
            if self.get_prediction(x) != nn.as_scalar(y):
                w.update(x, nn.as_scalar(y))
                mistake = True
        if mistake:
            return self.train(dataset)



class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.multiplier = -0.05
        self.batch_size = 100
        self.hidden_size = 300
        # what is input size???
        self.W1 = nn.Parameter(1, self.hidden_size)
        self.b1 = nn.Parameter(1, self.hidden_size)
        self.W2 = nn.Parameter(self.hidden_size, 1)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        # f(x) = relu(x * W1 + b1) * W2 + b2
        xW = nn.Linear(x, self.W1)
        predicted_y = nn.AddBias(xW, self.b1)
        predicted_y = nn.AddBias(nn.Linear(nn.ReLU(predicted_y), self.W2), self.b2)
        return predicted_y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        loss_scalar = float("inf")
        while loss_scalar > 0.015:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                grad_wrt_W1, grad_wrt_b1, grad_wrt_W2, grad_wrt_b2 = \
                    nn.gradients(loss, [self.W1, self.b1, self.W2, self.b2])
                self.W1.update(grad_wrt_W1, self.multiplier)
                self.b1.update(grad_wrt_b1, self.multiplier)
                self.W2.update(grad_wrt_W2, self.multiplier)
                self.b2.update(grad_wrt_b2, self.multiplier)
                loss_scalar = nn.as_scalar(loss)

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.multiplier = -0.05
        self.batch_size = 100
        self.hidden_size = 250
        # what is input size???
        self.W1 = nn.Parameter(784, self.hidden_size)
        self.b1 = nn.Parameter(1, self.hidden_size)
        self.W2 = nn.Parameter(self.hidden_size, 10)
        self.b2 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # f(x) = relu(x * W1 + b1) * W2 + b2
        xW = nn.Linear(x, self.W1)
        predicted_y = nn.AddBias(xW, self.b1)
        predicted_y = nn.AddBias(nn.Linear(nn.ReLU(predicted_y), self.W2), self.b2)
        return predicted_y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        accuracy = 0
        while accuracy < 0.9775:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                grad_wrt_W1, grad_wrt_b1, grad_wrt_W2, grad_wrt_b2 = \
                    nn.gradients(loss, [self.W1, self.b1, self.W2, self.b2])
                self.W1.update(grad_wrt_W1, self.multiplier)
                self.b1.update(grad_wrt_b1, self.multiplier)
                self.W2.update(grad_wrt_W2, self.multiplier)
                self.b2.update(grad_wrt_b2, self.multiplier)
                accuracy = dataset.get_validation_accuracy()