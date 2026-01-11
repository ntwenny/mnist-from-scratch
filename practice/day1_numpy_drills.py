import numpy as np
np.random.seed(0)

print("\n===== NUMPY EXAM =====\n")

# ------------------ 1. Array Creation ------------------
# a1: array [1,2,3,4]
# a2: 3x3 zeros matrix
# a3: numbers from 0 to 8 stepping by 2
# a4: 5 random numbers from uniform [0,1)

a1 = np.array([1, 2, 3, 4])
a2 = np.zeros((3, 3))
a3 = np.arange(0, 8, 2)
a4 = np.random.rand(5)


# ------------------ 2. Shapes ------------------
X = np.arange(24)

# r1: reshape X into (4,6)
# r2: flatten r1
# r3: transpose r1
# r4: total number of elements in r1

r1 = X.reshape(4, 6)
r2 = r1.flatten()
r3 = r1.T
r4 = r1.size


# ------------------ 3. Broadcasting ------------------
B = np.arange(12).reshape(4,3)

# B_norm: normalize each column: (B - mean)/std

B_norm = (B - B.mean(axis=0)) / B.std(axis=0)

# add_row: add [10, 20, 30] to each row of B

add_row = B + [10, 20, 30]

# mult_col: multiply each column of B by [1, 2, 3]

mult_col = B * [1, 2, 3]

# outer_add: add a column vector [100, 200, 300, 400] to B

col_vec = np.array([100, 200, 300, 400]).reshape(-1, 1)
outer_add = B + col_vec

# matrix_scale: multiply B by scalar 5 then add 3 to all elements

matrix_scale = B * 5 + 3


# ------------------ 4. Aggregations ------------------
# col_sum: sum of each column in B
# row_mean: mean of each row in B

col_sum = B.sum(axis=0)
row_mean = B.mean(axis=1)


# ------------------ 5. Indexing ------------------
C = np.array([-3,4,-1,2,0,5,-2])

# positive_vals: all values > 0
# slice_vals: elements from index 2 to 5

positive_vals = C[C > 0]
slice_vals = C[2:6]

# last_three: last 3 elements of C
# every_other: every other element starting from index 0

last_three = C[-3:]
every_other = C[0::2]


# ------------------ 6. Boolean Logic ------------------
# between: values in C that are between -1 and 3
# equals_two: True/False mask where C == 2

between = C[(C >= -1) & (C <= 3)]
equals_two = C == 2

# negative_or_big: values that are negative OR greater than 3
# not_zero: True/False mask where C is not equal to 0

negative_or_big = C[(C < 0) | (C > 3)]
not_zero = C != 0


# ------------------ 7. Vector Math ------------------
v1 = np.array([3,4])
v2 = np.array([1,2])

# dot_prod: dot product of v1 and v2
# norm_v1: Euclidean norm of v1

dot_prod = v1 @ v2
norm_v1 = np.linalg.norm(v1)


# ------------------ 8. Random ------------------
np.random.seed(1)

# rand_perm: random permutation of numbers 0–9
# rand_choice: choose 3 unique numbers from 0–9

rand_perm = np.random.permutation(10)
rand_choice = np.random.choice(10, 3, replace=False)


# ------------------ 9. Arg Functions ------------------
D = np.array([5,9,1,3,7])

# max_idx: index of max value in D
# where_big: indices where D > 4

max_idx = np.argmax(D)
where_big = np.where(D > 4)[0]

# min_idx: index of min value in D
# sorted_indices: indices that would sort D in ascending order

min_idx = np.argmin(D)
sorted_indices = np.argsort(D)


# ------------------ 10. Concatenation ------------------
M1 = np.ones((2,2))
M2 = np.zeros((2,2))

# stack_v: vertical stack M1 then M2
# stack_h: horizontal stack M1 then M2

stack_v = np.vstack((M1, M2))
stack_h = np.hstack((M1, M2))

print("\nFinished student file. Run numpy_check.py\n")