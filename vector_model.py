import math
import numpy as np
import pickle

class VectorModel:
    def __init__(self, layer_sizes, learning_rate=0.001, weight_decay=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.weights = []
        self.biases = []
        
        # AdamW parameters
        self.m_w = []
        self.v_w = []
        self.m_b = []
        self.v_b = []
        self.t = 0
        
        for i in range(len(layer_sizes) - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i+1]
            
            std = math.sqrt(2.0 / n_in)
            # weights[i] shape: (n_out, n_in)
            self.weights.append(np.random.normal(0, std, (n_out, n_in)))
            self.biases.append(np.zeros(n_out))
            
            self.m_w.append(np.zeros((n_out, n_in)))
            self.v_w.append(np.zeros((n_out, n_in)))
            self.m_b.append(np.zeros(n_out))
            self.v_b.append(np.zeros(n_out))

    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1.0, alpha)

    def softmax(self, x):
        max_x = np.max(x)
        exps = np.exp(x - max_x)
        return exps / np.sum(exps)

    def forward(self, x):
        """
        x: input vector (1D numpy array)
        """
        activations = [x]
        raw_values = []
        
        current_input = x
        for i in range(len(self.weights)):
            # "Vector-based": We treat each neuron's computation as a dot product
            # For speed, we use np.dot(W, x) which is effectively row-wise dot products
            z = np.dot(self.weights[i], current_input) + self.biases[i]
            raw_values.append(z)
            
            if i < len(self.weights) - 1:
                current_input = self.leaky_relu(z)
            else:
                current_input = self.softmax(z)
            activations.append(current_input)
            
        return activations, raw_values

    def backward(self, activations, raw_values, target_idx):
        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)
        
        y_pred = activations[-1]
        delta = y_pred.copy()
        delta[target_idx] -= 1.0
        
        for i in range(len(self.weights) - 1, -1, -1):
            layer_input = activations[i]
            
            # Gradient for bias and weights
            grads_b[i] = delta
            # outer product for weight gradient (vector * vectorT)
            # This is still vector-based in spirit as it's the result of scalar elements
            grads_w[i] = np.outer(delta, layer_input)
            
            if i > 0:
                # Delta for previous layer: W_transpose * current_delta
                delta = np.dot(self.weights[i].T, delta) * self.leaky_relu_derivative(raw_values[i-1])
                
        return grads_w, grads_b

    def update_params(self, grads_w, grads_b):
        self.t += 1
        for i in range(len(self.weights)):
            # Bias AdamW
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grads_b[i]
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (grads_b[i]**2)
            
            m_hat_b = self.m_b[i] / (1 - self.beta1**self.t)
            v_hat_b = self.v_b[i] / (1 - self.beta2**self.t)
            
            self.biases[i] -= self.learning_rate * (m_hat_b / (np.sqrt(v_hat_b) + self.epsilon))

            # Weight AdamW (Decoupled weight decay)
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grads_w[i]
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (grads_w[i]**2)
            
            m_hat_w = self.m_w[i] / (1 - self.beta1**self.t)
            v_hat_w = self.v_w[i] / (1 - self.beta2**self.t)
            
            self.weights[i] -= self.learning_rate * (
                m_hat_w / (np.sqrt(v_hat_w) + self.epsilon) + 
                self.weight_decay * self.weights[i]
            )

    def train_step(self, x, y):
        activations, raw_values = self.forward(x)
        grads_w, grads_b = self.backward(activations, raw_values, y)
        self.update_params(grads_w, grads_b)
        
        y_pred = activations[-1]
        loss = -np.log(max(y_pred[y], 1e-15))
        return loss

    def save_model(self, filepath):
        """
        Save the model state to a file.
        """
        state = {
            'layer_sizes': self.layer_sizes,
            'weights': self.weights,
            'biases': self.biases,
            'm_w': self.m_w,
            'v_w': self.v_w,
            'm_b': self.m_b,
            'v_b': self.v_b,
            't': self.t,
            'hyperparams': {
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'beta1': self.beta1,
                'beta2': self.beta2,
                'epsilon': self.epsilon
            }
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load_model(filepath):
        """
        Load the model state from a file and return a new VectorModel instance.
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        hp = state['hyperparams']
        model = VectorModel(
            layer_sizes=state['layer_sizes'],
            learning_rate=hp['learning_rate'],
            weight_decay=hp['weight_decay'],
            beta1=hp['beta1'],
            beta2=hp['beta2'],
            epsilon=hp['epsilon']
        )
        model.weights = state['weights']
        model.biases = state['biases']
        model.m_w = state['m_w']
        model.v_w = state['v_w']
        model.m_b = state['m_b']
        model.v_b = state['v_b']
        model.t = state['t']
        
        print(f"Model loaded from {filepath}")
        return model
