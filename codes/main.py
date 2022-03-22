import tensorflow as tf
import matplotlib.pyplot as plt
import _pickle as pickle
import numpy as np
from sklearn.model_selection import train_test_split


def read_data(file, num_sample, rc):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    repair_times = data[rc].loc[:, :num_sample].T
    initial_damage = repair_times.copy().replace({0, False}).astype('bool').astype('int')
    X_train, X_test, y_train, y_test = train_test_split(initial_damage, repair_times, random_state=0)
    return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy()/10, y_test.to_numpy()/10


X_train, X_test, y_train, y_test = read_data('../data/repair_times.pkl', 888, 0.01)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(500, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dense(500, activation='relu'),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=X_train.shape[1], activation="sigmoid")
])
model.summary()
# Untrained model
predictions = model(X_train).numpy()
loss_fn = tf.keras.losses.MeanAbsoluteError()
loss = loss_fn(y_train, predictions).numpy()

# # Define the Keras TensorBoard callback.
# logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)  # , callbacks=[tensorboard_callback])
model.evaluate(X_test, y_test, verbose=2)
y_pred = model.predict(X_test)