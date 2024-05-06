import pickle


def save(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)
