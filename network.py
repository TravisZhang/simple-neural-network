import numpy as np
import math
import math_utils


class Function:
    def __init__(self, type=None):
        self.type = type

    def sigmoid(self, x):
        return 1.0 / (1.0 + math.exp(-x))

    def deri_sigmoid(self, x):
        return x * (1.0 - x)

    def tanh(self, x):
        return math.tanh(x)

    def deri_tanh(self, x):
        return 1.0 - math.tanh(x)**2

    def softmax(self, x, i):
        return math.exp(x[i]) / sum([math.exp(xi) for xi in x])

    def forward(self, x):
        if self.type == 'sigmoid':
            return self.sigmoid(x)
        elif self.type == 'tanh':
            return self.tanh(x)
        return x

    def backward(self, x):
        if self.type == 'sigmoid':
            return self.deri_sigmoid(x)
        elif self.type == 'tanh':
            return self.deri_tanh(x)
        return 1.0


class Error:
    def __init__(self, type):
        self.type = type
        self.gradients = None
        self.errors = None

    def compute_error(self, y, t):
        assert (len(y) == len(t))
        if self.type == 'two-class':
            self.errors = np.array([
                -(ti * math.log(yi) + (1.0 - ti) * math.log(1.0 - yi))
                for yi, ti in zip(y, t)
            ])
        elif self.type == 'softmax':  # softmax/cross entropy error
            self.errors = np.array(
                [-(ti * math.log(yi)) for yi, ti in zip(y, t)])
        else:  # MSE
            self.errors = 0.5 * np.array([y - t])**2
        self.compute_gradient(y, t)
        return sum(self.errors)

    def compute_gradient(self, y, t):
        assert (len(y) == len(t))
        # de/dy
        if self.type == 'two-class':
            self.gradients = np.array([(yi - ti) / (yi * (1.0 - yi))
                                       for yi, ti in zip(y, t)])
        elif self.type == 'softmax':  # softmax/cross entropy error
            self.gradients = np.array([-(ti / yi) for yi, ti in zip(y, t)])
        else:  # MSE
            self.gradients = np.array([y - t])
            print('err grad:', self.gradients)


class Neuron:
    def __init__(self, parents, activation_function=None):
        self.parents = parents

        self.is_input_layer = self.parents is None
        self.activation_function = Function(activation_function)

        self.init_params()

    def set_weights(self, weights):
        self.weights = weights

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def init_params(self):
        self.bias = np.random.rand()
        self.bias = 0.0
        self.learning_rate = 0.1

        if self.is_input_layer:
            self.weights = 1.0
            self.grad_w = 0.0
            print('input layer!!!')
        else:
            self.weights = np.random.rand(len(self.parents))
            self.weights = np.ones(len(self.parents))
            self.weights = np.linspace(1, len(self.parents), len(self.parents))
            self.weights = np.random.normal(0.0,
                                            len(self.parents)**-0.5,
                                            len(self.parents))
            self.grad_w = np.zeros(len(self.parents))
            print('parents size:', len(self.parents))
        self.grad_b = 0.0

        self.input = 0.0
        self.output = 0.0
        self.calculate_output = True
        self.gradient_updated = False

    def init_grads(self):
        if self.is_input_layer:
            self.grad_w = 0.0
        else:
            self.grad_w = np.zeros(len(self.parents))
        self.grad_b = 0.0

    def enable_output_compute(self):
        self.calculate_output = True
        if self.is_input_layer:
            return
        [p.enable_output_compute() for p in self.parents]

    def set_input(self, input):
        self.input = input

    def get_output(self):
        self.gradient_updated = False
        self.init_grads()
        if self.is_input_layer:
            return self.input
        if self.calculate_output is True:
            input = np.array([parent.get_output() for parent in self.parents])
            input = np.squeeze(input)
            s = sum(input * self.weights) + self.bias
            self.output = self.activation_function.forward(s)
            # print('input:', input, 'weights:', self.weights, 's:', s, 'h:',
            #       self.output)
            # self.output = s
            self.calculate_output = False
        return self.output

    def update_gradient(self, error):
        if self.is_input_layer:
            return
        # de/dh * dh/ds
        recursive_part = error * self.activation_function.backward(self.output)
        recursive_part = np.squeeze(recursive_part)
        # de/dh * dh/ds * ds/dw
        update_w = np.array(
            [recursive_part * p.get_output() for p in self.parents])
        update_w = np.squeeze(update_w)
        # print('grad_w shape:', self.grad_w.shape, 'update_w shape:',
        #       update_w.shape)
        self.grad_w += update_w
        # print('grad_w:', self.grad_w, 'update_w:', update_w)
        # de/dh * dh/ds * ds/dw
        self.grad_b += recursive_part
        # de/dh * dh/ds * ds/dh'
        [
            p.update_gradient(recursive_part * w)
            for p, w in zip(self.parents, self.weights)
        ]
        self.gradient_updated = True

    def update_weights(self):
        if self.gradient_updated is False:
            return
        self.weights -= self.learning_rate * self.grad_w
        self.bias -= self.learning_rate * self.grad_b
        print('grad_w:', self.grad_w, 'grad_b:', self.grad_b)
        print('weights:', self.weights, 'bias:', self.bias)
        pass


class Layer:
    def __init__(self, neuron_num, activation_function, last_layer=None):
        self.last_layer = last_layer
        self.is_input_layer = self.last_layer is None
        self.activation_function = activation_function
        print('act fun:', self.activation_function)
        if self.is_input_layer:
            self.neurons = [
                Neuron(None, activation_function) for i in range(neuron_num)
            ]
        else:
            self.neurons = [
                Neuron(last_layer.neurons, activation_function)
                for i in range(neuron_num)
            ]
        self.output = None

    def set_hyper_params(self, learning_rate):
        [neuron.set_learning_rate(learning_rate) for neuron in self.neurons]

    def compute_output(self):
        self.output = np.array(
            [neuron.get_output() for neuron in self.neurons])
        if self.neurons[0].activation_function.type == 'softmax':
            print('compute_output softmax')
            exp_output = [math.exp(o) for o in self.output]
            exp_sum = sum(exp_output)
            self.output = np.array([exp_o / exp_sum for exp_o in exp_output])
        return self.output

    def set_input(self, input):
        assert (len(input) == len(self.neurons))
        [neuron.set_input(i) for neuron, i in zip(self.neurons, input)]

    def update_gradient(self, errors):
        if self.neurons[0].activation_function.type == 'softmax':
            gradients_e_sk = np.zeros(len(errors))
            for k, yk in enumerate(self.output):
                gradients_yi_sk = np.array([
                    yk * (1.0 - yk) if k == i else -yk * yi
                    for i, yi in enumerate(self.output)
                ])
                gradients_e_sk[k] = sum(gradients_yi_sk * errors)
            [
                neuron.update_gradient(gradients_e_sk[i])
                for i, neuron in enumerate(self.neurons)
            ]
        else:
            [
                neuron.update_gradient(errors[i])
                for i, neuron in enumerate(self.neurons)
            ]

        [neuron.enable_output_compute() for neuron in self.neurons]

    def update_weights(self):
        [neuron.update_weights() for neuron in self.neurons]


class DNNModel:
    def __init__(self, error_type=None):
        self.layers = []
        self.error_type = error_type
        self.output = 0.0
        self.error = 0.0

    def add_layer(self, neuron_num, activation_function=None):
        if len(self.layers) == 0:
            self.layers.append(Layer(neuron_num, activation_function, None))
        else:
            self.layers.append(
                Layer(neuron_num, activation_function, self.layers[-1]))

    def set_hyper_params(self, learning_rate):
        [[neuron.set_learning_rate(learning_rate) for neuron in layer.neurons]
         for layer in self.layers]

    def compute_output(self, input):
        assert (len(input) == len(self.layers[0].neurons))
        if len(self.layers) == 0:
            return None
        self.layers[0].set_input(input)
        self.output = self.layers[-1].compute_output()
        return self.output

    def init_network(self):
        [[neuron.init_params() for neuron in layer.neurons]
         for layer in self.layers]

    def train(self, inputs, batch):
        # update gradient
        self.error = 0.0
        for input, t in zip(inputs, batch):
            self.output = self.compute_output(input)
            print('input:', input, 't:', t, 'output:', self.output)
            err = Error(self.error_type)
            self.error += np.squeeze(
                err.compute_error(self.output, np.array([t])))
            self.layers[-1].update_gradient(err.gradients)
        # update weights
        for layer in self.layers:
            [neuron.update_weights() for neuron in layer.neurons]

    def test(self):
    	# Ref:
	# https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-bikesharing
        self.add_layer(3)
        self.add_layer(2, 'sigmoid')
        self.add_layer(1)

        inputs = np.array([[0.5, -0.2, 0.1]])
        targets = np.array([[0.4]])
        test_w_i_h = np.array([[0.1, -0.2], [0.4, 0.5], [-0.3, 0.2]])
        test_w_h_o = np.array([[0.3], [-0.1]])

        inputs = np.squeeze(inputs)
        targets = np.squeeze(targets)
        test_w_i_h = test_w_i_h.transpose()
        test_w_h_o = np.squeeze(test_w_h_o)

        self.set_hyper_params(0.5)

        [
            neuron.set_weights(weights)
            for weights, neuron in zip(test_w_i_h, self.layers[1].neurons)
        ]

        self.layers[2].neurons[0].set_weights(test_w_h_o)

        self.output = self.compute_output(inputs)
        print('test output:', self.output)

        err = Error(self.error_type)
        self.error += np.squeeze(
            err.compute_error(self.output, np.array([targets])))
        self.layers[-1].update_gradient(err.gradients)

        # update weights
        for layer in self.layers:
            [neuron.update_weights() for neuron in layer.neurons]


m = DNNModel()
m.add_layer(2)
# m.add_layer(3)
# m.add_layer(5, 'softmax')
m.add_layer(1)  # MSE

input = [1, 2]
output = m.compute_output(input)
print('output:', output)

# Create random input and output data
x = np.linspace(-math.pi, math.pi, 200)
y = np.sin(x)

iter_num = 2
batch_size = 10
m.init_network()
m.set_hyper_params(0.001)
errors = []
for i in range(iter_num):
    batch_indices = math_utils.kd_shuffle(len(x), batch_size)
    batch = np.array(y[batch_indices])
    # print('batch:', batch)
    inputs = np.array([x[batch_indices], x[batch_indices]**2])
    inputs = inputs.transpose()

    m.train(inputs, batch)
    print('iter:', i, 'error:', m.error)
    errors.append(m.error)
print('errors:', errors)

# n = DNNModel()
# n.test()
