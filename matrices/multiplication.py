import numpy as np

# make arrays
a = np.array([[4, 3, 2], [5, 6, 7]])
b = np.array([[6, 6, 6], [5, 5, 5]])

# make predictions calculated by myself
test_b_tran = np.array([[6, 5], [6, 5], [6, 5]])
test_c = np.array([[54, 45], [108, 90]])

# compute multiplication of matrices a and b
b_tran = b.transpose()  # b_tran: [[6, 5], [6, 5], [6, 5]]
c = a.dot(b_tran)  # c: [[ 54, 45], [108, 80]

# check the correctness
print(np.array_equal(b_tran, test_b_tran))
print(np.array_equal(c, test_c))


