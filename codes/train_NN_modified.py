import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split


def extract_repair_sequence(seq):
    """
    Extract the sequence of node repairs
    """
    sort_seq = seq.sort()
    repair_seq_order = sort_seq[0][sort_seq[0] != 0]
    repair_seq_nodes = sort_seq[1][sort_seq[0] != 0]
    return repair_seq_order, repair_seq_nodes


class Trainer:
    """
    Training the NN for the infrastructure data
    """

    def __init__(self, X, y, num_epoch, num_train, learning_rate, batch_size):
        self.num_epoch = num_epoch
        self.num_train = num_train
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.X = X
        self.y = y
        self.model = tf.keras.models.Sequential([tf.keras.layers.Dense(100, activation='relu',
                                                                       input_shape=(self.X.shape[1],)),
                                                 tf.keras.layers.Dense(100, activation='relu'),
                                                 tf.keras.layers.Dense(100, activation='relu'),
                                                 tf.keras.layers.Dense(100, activation='relu'),
                                                 tf.keras.layers.Dense(units=self.y.shape[1], activation="sigmoid")])

    def train(self):
        # Create list to collect loss for plot
        train_plot = []
        valid_plot = []

        # Training and validation split
        X_train, X_valid, y_train, y_valid = train_test_split(self.X, self.y, random_state=0)

        # Choose Optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        # Choose Loss Function
        loss_func = tf.keras.losses.MeanAbsoluteError()
        # Compile and fir the model
        self.model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy'])
        self.model.fit(X_train, y_train, epochs=self.num_epoch, batch_size=self.batch_size)

        # Run the updated NN model for train data
        train_predict = self.model(X_train).numpy()
        train_loss = loss_func(train_predict, y_train)
        # Run the updated NN model for validation data
        valid_predict = self.model(X_valid).numpy()
        valid_loss = loss_func(valid_predict, y_valid).numpy()

        # Append loss values for plot
        train_plot.append(train_loss.item())
        valid_plot.append(valid_loss.item())
        plt.figure()
        plt.plot(train_plot)
        plt.plot(valid_plot)
        plt.ylim(0, 0.003)
        plt.show()

        return self.model

    def test_data(self, X_test, y_test, accuracy):
        accuracy_index = []
        n, m = X_test.shape
        y_predict = self.model(X_test).detach()
        for i in range(n):
            counter = 0
            diff = tf.math.abs(21 * (y_predict[i].reshape(-1, 1) - y_test[i, :].reshape(-1, 1)))
            for j in range(m):
                if diff[j] <= accuracy and y_test[i, j].item() != 0:
                    counter += 1
            total = np.count_nonzero(y_test[i, :])

            if total == 0:
                accuracy_index.append(100)
            else:
                accuracy_index.append(counter / total * 100)
        print(f"Test accuracy {accuracy} was successfully done!")
        return accuracy_index
