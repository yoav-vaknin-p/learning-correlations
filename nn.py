import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
import torch
\from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import tensorly
from tensorly.decomposition import parafac
import matplotlib.pyplot as plt
import numpy as np

# Config

# Setting tensorly backend to work with Pytorch
tensorly.set_backend('pytorch')

## ---- Global learning parameters ---- ##

epochs_num = 2
mini_epoch_size = 50  # Measured in SGD steps
batch_size = 50
eta = 0.1  # Initial learning rate
model_act = F.relu  # Activation type for the neural net
num_of_images_per_mini_epoch = mini_epoch_size * batch_size  # num of images per mini-epoch
NUM_OF_LAYERS = 4

# Architecture, node numbers per layer
input_nodes_num = 28 * 28
hidden_nodes_num = 500
out_nodes_num = 10

# Data collectors for statistical purposes
train_errors, test_errors = [], []

##--  MNIST loading --##
train_dataset = datasets.MNIST(
    root='data',
    train=True,
    transform=ToTensor(),
    download=True,
)
test_dataset = datasets.MNIST(
    root='data',
    train=False,
    transform=ToTensor(),
)

train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size * mini_epoch_size, shuffle=True)


# Defining the NeuralNet
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Linear fully connected layers
        self.fc1 = nn.Linear(input_nodes_num, hidden_nodes_num, bias=True)
        self.fc2 = nn.Linear(hidden_nodes_num, hidden_nodes_num, bias=True)
        self.fc3 = nn.Linear(hidden_nodes_num, hidden_nodes_num, bias=True)
        self.fc4 = nn.Linear(hidden_nodes_num, out_nodes_num, bias=True)

        # Data aggregation members
        self.hidden_layer_1_data = tensor([])
        self.hidden_layer_2_data = tensor([])
        self.hidden_layer_3_data = tensor([])
        self.output_layer_data = tensor([])

    def forward(self, x):
        # We don't keep track of more than one forward call (Large memory cost & no need)
        model.hidden_layer_1_data = tensor([])
        model.hidden_layer_2_data = tensor([])
        model.hidden_layer_3_data = tensor([])
        model.output_layer_data = tensor([])

        # first layer pass
        out1 = self.fc1(x)
        act1 = model_act(out1)

        # first layer recording
        self.hidden_layer_1_data = out1

        # second layer pass
        out2 = self.fc2(act1)
        act2 = model_act(out2)

        # second layer recording
        self.hidden_layer_2_data = out2

        # third layer pass
        out3 = self.fc3(act2)
        act3 = model_act(out3)

        # third layer recording
        self.hidden_layer_3_data = out3

        output = self.fc4(act3)

        # output layer recording
        self.output_layer_data = output
        return output


## -- Train loop -- ##

def train(model, loss_func, optimizer):
    # Keeping track of the input samples we saw during training
    input_samples_collected_during_training, input_labels_collected_during_training =   tensor([]), tensor([])

    # Train and Test errors recording
    train_errors, test_errors = [], []

    # Train & Test activity pattern tensors
    train_layer_1_data, train_layer_2_data, train_layer_3_data, train_output_layer_overall_data = tensor([]), \
                                                                                                  tensor([]), \
                                                                                                  tensor([]), tensor([])

    test_layer_1_data, test_layer_2_data, test_layer_3_data, test_output_layer_overall_data = tensor([]), tensor([]), \
                                                                                              tensor([]), tensor([])

    for epoch in range(epochs_num):

        for step, (train_x, train_y) in enumerate(train_data_loader):

            # Flatting the input from (28,28) --> (728) vectors
            flatten_inp = torch.flatten(train_x, 1)

            # Keeping track of seen-in-training input vectors
            input_samples_collected_during_training = torch.cat((input_samples_collected_during_training, flatten_inp),
                                                                dim=0)
            input_labels_collected_during_training = torch.cat((input_labels_collected_during_training, train_y), dim=0)

            # forward pass #
            y_pred = model(flatten_inp)

            # Back pass (Backpropagation) #
            # No need to keep old gradients from prior backward() calls
            optimizer.zero_grad()

            # Calculate loss
            loss = loss_func(y_pred, train_y)

            # Calculate gradients
            loss.backward()

            # SGD step
            optimizer.step()

            # Every mini-epoch we get activity patterns & errors
            if (step + 1) % mini_epoch_size == 0:

                # Get num_of_images-per-mini-epoch random indices in range of (0, len(input samples seen in training))
                train_images_indices = torch.randperm(len(input_samples_collected_during_training))[
                                       :num_of_images_per_mini_epoch]

                # Use the random indices to select train data samples and their corresponding annotations
                train_x_data = torch.index_select(input_samples_collected_during_training, 0, train_images_indices)
                train_y_data = torch.index_select(input_labels_collected_during_training, 0, train_images_indices)

                # Getting random test data, X and y, batch size is images-per-mini-epoch
                unflattened_test_x_data, test_y_data = next(iter(test_data_loader))

                # Flatten the test_X data from (28,28) --> (784)
                test_x_data = torch.flatten(unflattened_test_x_data, 1)

                # get activity patterns for train and test data
                model(test_x_data.detach())
                test_layer_1_act, test_layer_2_act, test_layer_3_act, test_output = model.hidden_layer_1_data, \
                                                                                    model.hidden_layer_2_data, \
                                                                                    model.hidden_layer_3_data, \
                                                                                    model.output_layer_data

                model(train_x_data.detach())
                train_layer_1_act, train_layer_2_act, train_layer_3_act, train_output = model.hidden_layer_1_data, \
                                                                                        model.hidden_layer_2_data, \
                                                                                        model.hidden_layer_3_data, \
                                                                                        model.output_layer_data

                # Updating activity patterns globally to add current mini-epoch tensor
                # Train data updating
                train_layer_1_data = torch.cat((torch.unsqueeze(train_layer_1_act, dim=2), train_layer_1_data), dim=2)
                train_layer_2_data = torch.cat((torch.unsqueeze(train_layer_2_act, dim=2), train_layer_2_data), dim=2)
                train_layer_3_data = torch.cat((torch.unsqueeze(train_layer_3_act, dim=2), train_layer_3_data), dim=2)
                train_output_layer_overall_data = torch.cat(
                    (torch.unsqueeze(train_output, dim=2), train_output_layer_overall_data), dim=2)
                # Test data updating
                test_layer_1_data = torch.cat((torch.unsqueeze(test_layer_1_act, dim=2), test_layer_1_data), dim=2)
                test_layer_2_data = torch.cat((torch.unsqueeze(test_layer_2_act, dim=2), test_layer_2_data), dim=2)
                test_layer_3_data = torch.cat((torch.unsqueeze(test_layer_3_act, dim=2), test_layer_3_data), dim=2)
                test_output_layer_overall_data = torch.cat(
                    (torch.unsqueeze(test_output, dim=2), test_output_layer_overall_data), dim=2)

                # Error calc
                # Soft-max on the activation, will be used to achieve the errors
                pred_train_out, pred_test_out = torch.softmax(train_output, 1), torch.softmax(test_output, 1)

                # Append errors to relevant lists
                train_errors.append(error_calc(pred_train_out, train_y_data))
                test_errors.append(error_calc(pred_test_out, test_y_data))

                # Keep track on errors during the training
                print("train err - ", train_errors[-1], " test err - ", test_errors[-1])
        print("finished epoch number", epoch+1)

        # Done with epoch, new round begins --> reset the inputs-seen tracker
        input_samples_collected_during_training, input_labels_collected_during_training = tensor([]), tensor([])

    return (train_layer_1_data, train_layer_2_data, train_layer_3_data, train_output_layer_overall_data), \
           (test_layer_1_data, test_layer_2_data, test_layer_3_data, test_output_layer_overall_data), \
           (train_errors, test_errors)


# -- Statistical Pipelines -- #

def error_calc(y_hat, y):
    # In case we want to showcase   -Log(pred[real_label])
    # acc_err = 0
    # for sample_index in range(len(y_hat)):
    #     real_y = y[sample_index].int()
    #     acc_err += (math.log((y_hat[sample_index])[real_y]) * -1)
    # return acc_err / len(y_hat)

    # In case we want to showcase Accuracy
    correct_classification_count = 0
    for sample_index in range(len(y_hat)):
        real_y = y[sample_index].int()
        if (real_y == np.argmax(y_hat[sample_index].detach().numpy())): correct_classification_count += 1
    return 100 * (len(y) - correct_classification_count) / len(y)


def calculate_rank_err(factors, original_tensor, rank):
    estimated_tensor = np.zeros(original_tensor.shape)

    # Three dimensional cube decomposes to three factors groups
    factor1, factor2, factor3 = factors[0].detach().numpy(), factors[1].detach().numpy(), factors[2].detach().numpy()
    for factor_n in range(rank):
        # Summing the outer products of all factors-threesomes
        estimated_tensor += np.einsum('i,j,k', factor1[:, factor_n], factor2[:, factor_n], factor3[:, factor_n])
    # Return the reconstruction errors
    return np.mean(np.power(np.subtract(original_tensor, estimated_tensor), 2))


def statistical_pipeline(layer_tensor, layer_num):
    # Swap axes from (n, N, T) to have (T, N, n) tensor. T - mini epochs, N - neuron activity, n - images number
    base_tensor = tensorly.transpose(layer_tensor, (2, 1, 0))

    # Save activity pattern of last layer, DEBUGGING PURPOSE CODE
    # if layer_num == 3:
    #     k = 0
    #     for e in base_tensor:
    #         print("saving ", k)
    #         np.savetxt("data/Layer4Activations" + str(k) + ".txt",
    #                    e.detach().numpy(), delimiter=",  ")
    #         k += 1

    # Transposed tensor calc
    transposed_tensor = tensorly.transpose(base_tensor, (0, 2, 1))

    # Layer covariance tensor
    layer_cov_matrix = tensorly.matmul(base_tensor, transposed_tensor) / (base_tensor.size()[-1])  # images num

    # Saving the tensor-reconstruction errors of CANDECOMP / Parafac / Canonical Polyadic Decomposition
    reconstruction_errs = []
    for rank in range(1, 5):
        # Tensorly's parafac decomposition
        parafac_decomposition = parafac(layer_cov_matrix, rank=rank, svd='truncated_svd', return_errors=True)

        # the reconstruction error of this specific rank
        cur_rank_err = calculate_rank_err(parafac_decomposition[0].factors, layer_cov_matrix.detach().numpy(), rank)

        # keep in array to save later into a file
        reconstruction_errs.append(cur_rank_err)

        # Saving factors to .txt files
        for factor_num in range(len(parafac_decomposition[0].factors)):
            factor = (parafac_decomposition[0].factors)[factor_num]
            np.savetxt("data/Layer" + str(layer_num + 1) + "/rank" + str(rank) + "/factor" + str(factor_num) + ".txt",
                       factor.detach().numpy(), delimiter=",  ")

    # Saving reconstruction errors to .txt file
    np.savetxt("data/Layer" + str(layer_num + 1) + "/rank_errors.txt", np.array(reconstruction_errs), fmt='%s',
               delimiter=", ")


def plot_errors(errors):
    # Plot the train and test errors side by side
    train_errors, test_errors = errors
    x = [i for i in range(1, len(train_errors) + 1)]
    plt.plot(x, train_errors)
    plt.plot(x, test_errors)
    plt.xlabel("SGD steps (x" + str(mini_epoch_size) + ")")
    plt.ylabel("Misclassified examples (%)")
    plt.legend(["Train error", "Test error"])
    plt.show()


if __name__ == '__main__':
    # Initiating the custom model
    model = NeuralNetwork()

    # Defining the loss function
    loss_func = nn.CrossEntropyLoss()

    # SGD optimizer with chosen learning rate
    optimizer = optim.SGD(model.parameters(), lr=eta)

    # get train and test activation, errors list from train pipeline
    train_act_tensor, test_act_tensor, errors = train(model, loss_func,
                                                      optimizer)  # Holds layer 1-3, output data from the train phase
    # Plot errors
    plot_errors(errors)

    # Statistical analysis for the activity patterns
    for layer_num in range(NUM_OF_LAYERS):
        print("Statistical Analysis of Layer ", layer_num + 1)
        # Passing Layer_act-mean(Layer_act). (n, N, T) where T = num of mini-batches, n - num of images per mini-batch, N - num of neurons
        statistical_pipeline(test_act_tensor[layer_num] - torch.mean(test_act_tensor[layer_num], 0, keepdim=True),
                             layer_num)
        print("End of Statistical Analysis of Layer ", layer_num + 1)

    # Visualization pipeline
    for layer_num in range(NUM_OF_LAYERS):
        # Load the rank reconstruction errors (saved during stat' analysis) and select the best rank
        rank_reconstruction_errors = np.loadtxt("data/Layer" + str(layer_num + 1) + "/rank_errors.txt", delimiter=", ")
        best_rank = np.argmin(rank_reconstruction_errors) + 1
        # Load the time factor & correlation factor
        time_factor = np.loadtxt("data/Layer" + str(layer_num + 1) + "/rank" + str(best_rank) + "/factor0.txt",
                                 delimiter=", ")[:-1]  # last step is not full
        cov_factor = np.loadtxt("data/Layer" + str(layer_num + 1) + "/rank" + str(best_rank) + "/factor1.txt",
                                delimiter=", ")

        # Plotting
        graph_colors = [
            "goldenrod", "tomato", "orange", "slateblue"
        ]

        fig, axs = plt.subplots(best_rank, 2)
        fig.suptitle("Layer " + str(layer_num + 1) + " Factors" if layer_num != 3 else "Output Layer Factors")
        X_bar = [i for i in range(cov_factor.shape[0])]  # Bar plot X values
        X_plot = [i for i in range(time_factor.shape[0])]  # Regular plot X valuse
        plot_num = 0
        for row in range(best_rank):
            for col in range(2):
                # Plotting the relevant graphs side by side
                axs[row][0].bar(X_bar, cov_factor[:, plot_num], 0.9, color=graph_colors[plot_num])
                axs[row][1].plot(X_plot, time_factor[:, plot_num], color=graph_colors[plot_num])

                # Bar plot specifics
                axs[row][0].axhline(0, color='grey', linewidth=0.8)
                if layer_num == 3:
                    axs[row][0].set_xticks(X_bar)

                # Set super-titles for plots
                if not row:
                    axs[row][0].set_title("Covariance Factors")
                    axs[row][1].set_title("Time Factors")

            plot_num += 1
        plt.show()

    # Analyzing the output layer activity tensor
    for i in range(48):  # 48 mini-epochs = Two full epochs
        # Current mini-epoch matrix, size - (N, n) where N - Neurons num, n - images num
        cur_batch = np.loadtxt("data/Layer4Activations" + str(i) + ".txt", delimiter=",")
        # Print variance list, where [first neuron var, second neuron var,...,10th neuron var]
        print("Variance for batch ", i, ": ", np.var(cur_batch, axis=1))

        # Plot start, middle, end distributions
        if i in (0, 24, 47):
            fig, axs = plt.subplots(2, 5)
            for k in range(10):
                n_bins = 200
                axs[k // 5][k % 5].hist(cur_batch[k, :], bins=n_bins)
            fig.show()
            plt.show()
