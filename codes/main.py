import tensorflow as tf
import numpy as np

# Initialization #################################
load_train_data = 0
load_test_data = 0

train_NN = 0
train_NN_encode = 0

rsc_allocation = 1

NN_params = 0
params_visual = 0
params_encode = 0
save_NN_params = 0

test_results = 0

visualize = 0

tf.random.set_seed(2)

# Load Data ####################################
if load_train_data:
    from data_loader import load

    # "X" and "Y" are Nxn matrices where "N" is the number of scenarios and "n" is the number of nodes. Each row of
    # "X" is a binary vector which has a "0" when the node damaged and "1" when the node is repaired. Each element
    # of "Y" gives the time-step at which the node is repaired and "0" if the node is not damaged.
    DATA = load(horizon=21, num_nodes=125, num_layers=3, num_rsc=7)
    X_train, y_train = DATA.read_train(num_sample=10000)
    X_train = 1 - X_train
    print("\nTraining data was successfully loaded!\n")

if load_test_data:
    from data_loader import load

    DATA = load(horizon=21, num_nodes=125, num_layers=3, num_rsc=7)
    X_test, y_test = DATA.read_test(mags=[7], num_scenario=1000)
    X_test = 1 - X_test  # Now "0" implies a repaired node, "1" implies damaged node
    print("\nTest data was successfully loaded!\n")

# Train Neural Network #############################
if train_NN:
    from train_NN_modified import Trainer

    trainer = Trainer(X_train, y_train, num_epoch=1000, num_train=9000, learning_rate=0.01, batch_size=32)
    model = trainer.train()  # The output model contains all features and parameters of the Neural Network
    print("\nModel was successfully trained!\n")

# Resource Allocation ###############################
if rsc_allocation:
    from data_loader import load
    from train_NN_modified import Trainer
    import matplotlib.pyplot as plt

    max_true = []
    max_learned = []

    for i in range(2, 8):
        print(f"\nResource No. {i} Loading...\n")
        # Read train data
        DATA = load(horizon=21, num_nodes=125, num_layers=3, num_rsc=i)
        X_train, y_train = DATA.read_train(num_sample=10000)
        X_train = 1 - X_train
        print(f"\nTraining data was successfully loaded!")
        # Read test data corresponding
        X_test, y_test = DATA.read_test(mags=[6, 7, 8, 9], num_scenario=470)
        X_test = 1 - X_test
        print(f"Test data was successfully loaded!")
        # Training NN
        trainer = Trainer(X_train, y_train, num_epoch=1000, num_train=9500, learning_rate=0.01, batch_size=32)
        model = trainer.train()
        print("\nModel was successfully trained!\n")
        # Test and save results
        y_predict = tf.math.round(21 * model.predict(X_test))
        max_true.append(21 * tf.math.reduce_max(y_test))
        max_learned.append(tf.math.reduce_max(y_predict).item())

    interval = range(2, 8)
    plt.plot(interval, max_learned, marker='o', color='r', linestyle='-', linewidth=2)
    plt.plot(interval, max_true, marker='o', color='k', linestyle='-', linewidth=0.5)

    for i in range(7):
        plt.vlines(x=i + 2, ymin=5, ymax=max_true[i], linestyle='--', linewidth=0.3)

    plt.vlines(x=2, ymin=5, ymax=max_true[0], linestyle='--', linewidth=1)
    plt.vlines(x=3, ymin=5, ymax=max_true[1], linestyle='--', linewidth=1)
    plt.vlines(x=4, ymin=5, ymax=max_true[2], linestyle='--', linewidth=1)
    plt.vlines(x=5, ymin=5, ymax=max_true[3], linestyle='--', linewidth=1)
    plt.vlines(x=6, ymin=5, ymax=max_true[4], linestyle='--', linewidth=1)
    plt.vlines(x=7, ymin=5, ymax=max_true[5], linestyle='--', linewidth=1)
    plt.vlines(x=8, ymin=5, ymax=max_true[6], linestyle='--', linewidth=1)

    plt.xlim(1.8, 7.2)
    plt.ylim(5, 19)

    plt.legend(['Predicted Recovery Time', 'td-INDP Recovery Time'])
    plt.xlabel('Number of Resources')
    plt.ylabel('Recovery Time')
    plt.savefig('ResourceAllocation.png', dpi=1000)
    plt.show()

# Extract Neural Net Params ############################
if NN_params:
    import matplotlib.pyplot as plt
    import numpy as np

    num_hidden = 1
    num_nodes = 125
    W = []
    b = []
    for i in range(num_hidden + 1):
        W.append(list(model.getWeights())[2 * i].detach())
        b.append(list(model.getWeights())[2 * i + 1].detach())
    print("\nModel parameters were successfully obtained!\n")

    for i in range(num_hidden):
        np.savetxt("W" + str(i + 1), W[i], delimiter=",")
        np.savetxt("b" + str(i + 1), b[i], delimiter=",")
    print("\nModel parameters were successfully saved!\n")

    if params_visual:
        interaction_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                interaction_matrix[i, j] = tf.tensordot(W[0][:, i], W[1][j, :])
        threshold = 0.2
        interaction_matrix[np.abs(interaction_matrix) < threshold] = 0
        interaction_matrix = interaction_matrix.T
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(6, 6)
        cax = ax.matshow(np.abs(interaction_matrix), vmin=0, vmax=1.1, interpolation='none', cmap=plt.get_cmap('PuBu'))
        fig.colorbar(cax)
        plt.vlines(x=49, ymin=0, ymax=124, color='red', linestyle='-', linewidth=3)
        plt.vlines(x=65, ymin=0, ymax=124, color='red', linestyle='-', linewidth=3)
        plt.hlines(y=49, xmin=0, xmax=124, color='red', linestyle='-', linewidth=3)
        plt.hlines(y=65, xmin=0, xmax=124, color='red', linestyle='-', linewidth=3)

        plt.xlabel("\nPower" + 14 * " " + "Gas" + 21 * " " + "Water")
        plt.ylabel(5 * " " + "Power" + 24 * " " + "Gas" + 20 * " " + "Water")

        fig.savefig('Matrix.png', dpi=1000)
        plt.show()
