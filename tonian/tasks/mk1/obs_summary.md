# Observation and Action Summary of the mk1 Robot

# Action Tensor
    - 0. left_sholder_a
    - 1. left_sholder_b
    - 2. left_arm_rotate
    - 3. left_elbow
    - 4. right_sholder_a
    - 5. right_sholder_b
    - 6. right_arm_rotate
    - 7. right_elbow
    - 8. torso
    - 9. left_hip_a
    - 10. left_hip_b
    - 11. left_knee
    - 12. left_foot
    - 13. right_hip_a
    - 14. right_hip_b
    - 15. right_knee
    - 16. right_foot

# Torch action 1:

 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16
 1  0  1  0  1  1  1  1  0  1  1  0  1  0  0  1  1
 1  0  1  0  1  1  1  1  0  1  1  0  1  0  0  1  1
 1  0  1  0  1  1  1  1  0  1  1  0  1  0  0  1  1
 1  0  1  0  1  1  1  1  0  1  1  0  1  0  0  1  1
 1  0  1  0  1  1  1  1  0  1  1  0  1  0  0  1  1
 1  0  1  0  1  1  1  1  0  1  1  0  1  0  0  1  1
 1  0  1  0  1  1  1  1  0  1  1  0  1  0  0  1  1
 1  0  1  0  1  1  1  1  0  1  1  0  1  0  0  1  1
 1  0  1  0  1  1  1  1  0  1  1  0  1  0  0  1  1
 1  0  1  0  1  1  1  1  0  1  1  0  1  0  0  1  1
[ CUDAFloatType{10,17} ]
Lower Limit Tensor
 0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0
[ CUDAFloatType{10,17} ]


# Torch action -1

Upper Limit Tensor
 0  0  0  0  0  0  0  0 -1  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0 -1  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0 -1  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0 -1  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0 -1  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0 -1  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0 -1  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0 -1  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0 -1  0  0  0  0  0  0  0  0
 0  0  0  0  0  0  0  0 -1  0  0  0  0  0  0  0  0
[ CUDAFloatType{10,17} ]
Lower Limit Tensor
-1 -1 -1 -1  0 -1 -1 -1  0 -1 -1 -1 -1  0 -1 -1 -1
-1 -1 -1 -1  0 -1 -1 -1  0 -1 -1 -1 -1  0 -1 -1 -1
-1 -1 -1 -1  0 -1 -1 -1  0 -1 -1 -1 -1  0 -1 -1 -1
-1 -1 -1 -1  0 -1 -1 -1  0 -1 -1 -1 -1  0 -1 -1 -1
-1 -1 -1 -1  0 -1 -1 -1  0 -1 -1 -1 -1  0 -1 -1 -1
-1 -1 -1 -1  0 -1 -1 -1  0 -1 -1 -1 -1  0 -1 -1 -1
-1 -1 -1 -1  0 -1 -1 -1  0 -1 -1 -1 -1  0 -1 -1 -1
-1 -1 -1 -1  0 -1 -1 -1  0 -1 -1 -1 -1  0 -1 -1 -1
-1 -1 -1 -1  0 -1 -1 -1  0 -1 -1 -1 -1  0 -1 -1 -1
-1 -1 -1 -1  0 -1 -1 -1  0 -1 -1 -1 -1  0 -1 -1 -1