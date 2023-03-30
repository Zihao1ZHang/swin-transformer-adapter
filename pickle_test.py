import pickle

if __name__ == "__main__":
    with open("./data_for_plot/190895_16.pickle", 'rb') as handle:
        b = pickle.load(handle)
    print(b)
