import numpy as np
import day1_numpy_drills as s

def check(name, val, exp):
    ok = np.allclose(val, exp)
    print(f"{name}: {'PASS' if ok else 'FAIL'}")

check("a1", s.a1, np.array([1,2,3,4]))
check("a2", s.a2, np.zeros((3,3)))
check("a3", s.a3, np.arange(0,9,2))
check("a4", s.a4.shape, (5,))

check("r1", s.r1.shape, (4,6))
check("r4", s.r4, 24)

B = np.arange(12).reshape(4,3)

# Broadcasting tests
check("B_norm", s.B_norm, (B - B.mean(axis=0)) / B.std(axis=0))
check("add_row", s.add_row, B + np.array([10, 20, 30]))
check("mult_col", s.mult_col, B * np.array([1, 2, 3]))
check("outer_add", s.outer_add, B + np.array([[100], [200], [300], [400]]))
check("matrix_scale", s.matrix_scale, B * 5 + 3)

check("col_sum", s.col_sum, B.sum(axis=0))
check("row_mean", s.row_mean, B.mean(axis=1))

C = np.array([-3,4,-1,2,0,5,-2])
check("positive_vals", s.positive_vals, C[C>0])
check("slice_vals", s.slice_vals, C[2:5])
check("last_three", s.last_three, C[-3:])
check("every_other", s.every_other, C[::2])
check("negative_or_big", s.negative_or_big, C[(C < 0) | (C > 3)])
check("not_zero", s.not_zero, C != 0)

check("dot_prod", s.dot_prod, 11)
check("norm_v1", s.norm_v1, 5)

D = np.array([5,9,1,3,7])
check("min_idx", s.min_idx, np.argmin(D))
check("sorted_indices", s.sorted_indices, np.argsort(D))

print("\nAll checks complete.")