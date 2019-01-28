import numpy as np

# make arrays
a = np.array([[4, 3, 2], [5, 6, 7]])
b = np.array([[6, 6, 6], [5, 5, 5]])
c = np.array([1, 2, 3])
# compute multiplication of matrices a and b
b_tran = b.transpose()  # b_tran: [[6, 5], [6, 5], [6, 5]]
multiplied = a.dot(b_tran)  # c: [[ 54, 45], [108, 80]
squared = np.power(c, 2)
# make predictions calculated by myself
test_b_tran = np.array([[6, 5], [6, 5], [6, 5]])
test_multiplied = np.array([[54, 45], [108, 90]])

# check the correctness
print(np.array_equal(b_tran, test_b_tran))
print(np.array_equal(multiplied, test_multiplied))

# https://github.com/stuartzong/integration


