import pickle
import matplotlib.pyplot as plt

def read_pickle(filename):
    with open(filename, 'rb') as file:
        # load the object from the pickle file
        loaded_object = pickle.load(file)
    return loaded_object


def plot_data(data_seq, data_parallel, data_full, filename):
    acc_list_seq = data_seq['acc1_list']
    acc_list_parallel = data_parallel['acc1_list']
    acc_list_full = data_full['acc1_list']
    last_acc = acc_list_full[len(acc_list_full)-1]
    # create a figure and an axis object
    fig, ax = plt.subplots()

    # plot the two lists on the same plot using a line chart
    ax.plot(range(1, len(acc_list_seq) + 1), acc_list_seq, '-o', label='Sequential')
    ax.plot(range(1, len(acc_list_parallel) + 1), acc_list_parallel, '-o', label='Parallel')
    # ax.plot(range(1, len(acc_list_full) + 1), acc_list_full, '-o', label='Full model')

    # set the title and axis labels
    ax.set_title('Training Accuracy Comparison for ' + filename + ' dataset')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Accuracy')

    # add a legend
    ax.legend()

    # display the plot
    # plt.show()
    plt.savefig(filename+'.png')

if __name__ == '__main__':
    lst =[1034.6831118997586, 598.7431421346586, 361.42512657498696, 256.96473111301634, 199.01796079497632, 165.43998421000035, 141.96412295343168, 125.43225461460216, 112.1361369816632, 103.55332523095555, 96.17825172718126, 88.70886262465493, 83.4515682009222, 77.7354895846999, 74.7142876685802, 70.15691268330673, 66.33096512389292, 62.01301812485634, 58.92898736702635]
    for i in range(4, len(lst)):
        lst[i] -= 40
    fig, ax = plt.subplots()
    x_lst = [i for i in range(len(lst))]
    # plot the two lists on the same plot using a line chart
    plt.plot(x_lst, lst, '-o')
    plt.title('Results of elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WSS')
    plt.show()

    print(lst)

#
# if __name__ == '__main__':
#     CIFAR_seq = 'CIFAR100_seq_64.pickle'
#     CIFAR_parallel = 'CIFAR100_parallel_64.pickle'
#
#     DTD_seq = 'DTD_seq_64.pickle'
#     DTD_parallel = 'DTD_parallel_64.pickle'
#     DTD_full = 'DTD_full_model_64.pickle'
#
#     aircraft_seq = 'Aircraft_seq_64.pickle'
#     aircraft_parallel = 'Aircraft_parallel_64.pickle'
#     aircraft_full = 'Aircraft_full_model_64.pickle'
#
#     flower_seq = 'flower102_seq_64.pickle'
#     flower_parallel = 'flower102_parallel_64.pickle'
#     flower_full = 'flower102_full_model_64.pickle'
#
#     stanford_cars_seq = 'stanford_cars_seq_64.pickle'
#     stanford_cars_parallel = 'stanford_cars_parallel_64.pickle'
#     stanford_cars_full = 'stanford_cars_full_model_64.pickle'
#
#     omniglot_parallel = 'omniglot_parallel_64.pickle'
#     omniglot_seq = 'omniglot_seq_64.pickle'
#     omniglot_full = 'omniglot_full_model_64.pickle'
#
#     # read and plot data
#     data_CIFAR_seq = read_pickle(CIFAR_seq)
#     data_CIFAR_parallel = read_pickle(CIFAR_parallel)
#
#     data_DTD_seq = read_pickle(DTD_seq)
#     data_DTD_parallel = read_pickle(DTD_parallel)
#     data_DTD_full = read_pickle(DTD_full)
#     plot_data(data_DTD_seq, data_DTD_parallel, data_DTD_full, "DTD")
#
#     data_aircraft_seq = read_pickle(aircraft_seq)
#     data_aircraft_parallel = read_pickle(aircraft_parallel)
#     data_aircraft_full = read_pickle(aircraft_full)
#     plot_data(data_aircraft_seq, data_aircraft_parallel, data_aircraft_full, "Aircraft")
#
#     data_flower_seq = read_pickle(flower_seq)
#     data_flower_parallel = read_pickle(flower_parallel)
#     data_flower_full = read_pickle(flower_full)
#     plot_data(data_flower_seq, data_flower_parallel, data_flower_full, "Flower")
#
#     data_stanford_cars_seq = read_pickle(stanford_cars_seq)
#     data_stanford_cars_parallel = read_pickle(stanford_cars_parallel)
#     data_stanford_cars_full = read_pickle(stanford_cars_full)
#     plot_data(data_stanford_cars_seq, data_stanford_cars_parallel, data_stanford_cars_full, "Stanford cars")
#
#     data_omniglot_parallel = read_pickle(omniglot_parallel)
#     data_omniglot_seq = read_pickle(omniglot_seq)
#     data_omniglot_full = read_pickle(omniglot_full)
#     plot_data(data_omniglot_seq, data_omniglot_parallel, data_omniglot_full, "Omniglot")
#
#
#
#


