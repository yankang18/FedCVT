import numpy as np


# def stable_softmax(X):
#     exps = np.exp(X - np.max(X, axis=1))
#     return exps / np.sum(exps, axis=1)

def softmax(x):
    """Compute the softmax of vector x."""
    exps = np.exp(x)
    print("exps", exps)
    print("np.sum(exps, axis=1):", np.sum(exps, axis=1))
    return exps / np.sum(exps, axis=1, keepdims=True)


def cross_entropy(p, y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    	Note that y is not one-hot encoded vector.
    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    print(1.47057772e+02)
    m = y.shape[0]
    # We use multidimensional array indexing to extract
    # softmax probability of the correct label for each sample.
    # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
    log_likelihood = -np.log(p[range(m), y])
    print("1 log_likelihood {0}".format(log_likelihood))
    loss = np.sum(log_likelihood) / m
    return loss


def cross_entropy_2(p, y):
    """
    X is the output from a softmax layer (num_examples x num_classes)
    y is labels (num_examples x num_classes). Note that y is one-hot encoded vector.
    """
    m = y.shape[0]
    log_likelihood = -np.log(p) * y
    print("2 log_likelihood {0}".format(log_likelihood))
    loss = np.sum(log_likelihood) / m
    return loss


if __name__ == "__main__":

    X = np.array([[3, 12, 7],
                  [7, 2, 1],
                  [4, 8, 11]])
    # y = np.array([[1],
    #               [0],
    #               [2]])

    y = np.array([1, 0, 2])

    print("y shape {0}".format(y.shape))

    y_2 = np.array([[0, 1, 0],
                    [1, 0, 0],
                    [0, 0, 1]])

    print("X {0}".format(X))
    print("y {0}".format(y))

    p = softmax(X)
    print("p: {0}".format(p))
    print("p: {0}".format(np.sum(p, axis=1)))

    loss = cross_entropy(p, y)
    print("loss {0}".format(loss))
    loss = cross_entropy_2(p, y_2)
    print("loss {0}".format(loss))
