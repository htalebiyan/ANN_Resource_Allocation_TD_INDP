import numpy as np
import csv
from zipfile import ZipFile as zf
import codecs


class load:
    """
    This class load the input data and generate feature and target matrices
    """
    def __init__(self, horizon, num_nodes, num_layers, num_rsc):
        self.horizon = horizon
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.num_rsc = num_rsc

    def read_train(self, num_sample):
        """
        Reading train data
        """
        X = np.zeros(shape=(self.num_nodes, num_sample))
        y = np.zeros(shape=(self.num_nodes, num_sample))
        # "Data" collects all of the data for different earthquake magnitudes and scenarios in one big matrix
        data = np.zeros(shape=(self.num_nodes, self.horizon, num_sample))
        file1 = '../tdINDP-Custom/td-INDP-Controlling-td-INP/results/tdindp_results_L3_m0_v' + \
                str(self.num_rsc) + '.zip'
        zip_file = zf(file1)

        for sample in range(num_sample):
            data = np.zeros(shape=(1, self.horizon))
            for layer in range(self.num_layers):
                file2 = 'tdindp_results_L3_m0_v' + str(self.num_rsc) + '/matrices/func_matrix_' + str(
                    sample + 10001) + '_layer_' + str(layer + 1) + '_.csv'
                read_file = zip_file.open(file2)
                readCSV = csv.reader(codecs.iterdecode(read_file, 'utf-8'))
                x = list(readCSV)
                n = len(x)
                for j in range(n):
                    data = np.concatenate((data, np.array(x[j]).reshape(1, self.horizon)), axis=0)
            data[:, :, sample] = data[1:, :]

            # Creating the input matrices
            X[:, sample] = data[:, 0, sample]

            # Creating the output matrices
            for node in range(self.num_nodes):
                if data[node, -1, sample] == 0:
                    y[node, sample] = 0
                else:
                    y[node, sample] = data[node, :, sample].argmax()

            if sample % 1000 == 0:
                print(f"{sample}/{num_sample} training data loaded")

        y = y / self.horizon  # Output Normalization
        return X, y

    def read_test(self, mags, num_scenario):
        """
        Reading test data based corresponding to different earthquake magnitudes scenarios for Shelby County
        """
        X = np.zeros(shape=(self.num_nodes, num_scenario * len(mags)))
        y = np.zeros(shape=(self.num_nodes, num_scenario * len(mags)))
        data = np.zeros(shape=(self.num_nodes, self.horizon, num_scenario, len(mags)))

        for magnitude in mags:
            file1 = '../INDP-Data/tdINDP-T20-V' + str(self.num_rsc) + '-W3-M' + str(magnitude) + '.zip'
            zip_file = zf(file1)
            m = mags.index(magnitude)
            for scenario in range(num_scenario):
                data = np.zeros(shape=(1, self.horizon))
                for layer in range(self.num_layers):
                    file2 = 'tdINDP-T20-V' + str(self.num_rsc) + '-W3-M' + str(magnitude) + '/func_matrix_' + str(
                        scenario + 1) + '_layer_' + str(layer + 1) + '_.csv'
                    read_file = zip_file.open(file2)
                    readCSV = csv.reader(codecs.iterdecode(read_file, 'utf-8'))
                    x = list(readCSV)
                    n = len(x)
                    for j in range(n):
                        data = np.concatenate((data, np.array(x[j]).reshape(1, self.horizon)), axis=0)
                data[:, :, scenario, m] = data[1:, :]

                # Creating the input matrices
                X[:, (m * num_scenario) + scenario] = data[:, 0, scenario, m]

                # Creating the output matrices
                for node in range(self.num_nodes):
                    if data[node, -1, scenario, m] == 0:
                        y[node, (m * num_scenario) + scenario] = 0
                    else:
                        y[node, (m * num_scenario) + scenario] = data[node, :, scenario, m].argmax()
        y = y / self.horizon
        return X, y
