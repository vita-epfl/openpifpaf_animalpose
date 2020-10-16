
import numpy as np

CATEGORIES = ('cat', 'cow', 'dog', 'sheep', 'horse')

ANIMAl_KEYPOINTS = [
    'Nose',         # 1
    'L_eye',        # 2
    'R_eye',        # 3
    'L_ear',        # 4
    'R_ear',        # 5
    'Throat',       # 6
    'Tail',         # 7
    'withers',      # 8
    'L_F_elbow',    # 9
    'R_F_elbow',    # 10
    'L_B_elbow',    # 11
    'R_B_elbow',    # 12
    'L_F_knee',     # 13
    'R_F_knee',     # 14
    'L_B_knee',     # 15
    'R_B_knee',     # 16
    'L_F_paw',      # 17
    'R_F_paw',      # 18
    'L_B_paw',      # 19
    'R_B_paw',      # 20
]


ALTERNATIVE_NAMES = [
    'Nose',         # 1
    'L_Eye',        # 2
    'R_Eye',        # 3
    'L_EarBase',    # 4
    'R_EarBase',    # 5
    'Throat',       # 6
    'TailBase',     # 7
    'Withers',      # 8
    'L_F_Elbow',    # 9
    'R_F_Elbow',    # 10
    'L_B_Elbow',    # 11
    'R_B_Elbow',    # 12
    'L_F_Knee',     # 13
    'R_F_Knee',     # 14
    'L_B_Knee',     # 15
    'R_B_Knee',     # 16
    'L_F_Paw',      # 17
    'R_F_Paw',      # 18
    'L_B_Paw',      # 19
    'R_B_Paw',      # 20
]


ANIMAL_SKELETON = [
    (1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 8), (4, 6), (8, 9), (8, 6), (5, 6), (5, 8), (8, 10), (9, 10),
    (9, 13), (13, 17), (10, 14), (14, 18), (8, 11), (8, 12), (11, 7), (12, 7), (11, 15), (15, 19), (12, 16), (16, 20)
]

ANIMAL_SIGMAS = [
    0.026,  # nose
    0.025,  # eyes
    0.025,  # eyes
    0.035,  # ears
    0.035,  # ears
    0.079,  # throat
    0.079,  # tail
    0.079,  # withers
    0.072,  # elbows
    0.072,  # elbows
    0.072,  # elbows
    0.072,  # elbows
    0.087,  # knees
    0.087,  # knees
    0.087,  # knees
    0.087,  # knees
    0.089,  # ankles
    0.089,  # ankles
    0.089,  # ankles
    0.089,  # ankles
]

split, error = divmod(len(ANIMAl_KEYPOINTS), 4)
ANIMAL_SCORE_WEIGHTS = [5.0] * split + [3.0] * split + [1.0] * split + [0.5] * split + [0.1] * error
assert len(ANIMAL_SCORE_WEIGHTS) == len(ANIMAl_KEYPOINTS)


ANIMAL_UPRIGHT_POSE = np.array([
    [0.0, 4.3, 2.0],  # 'nose',            # 1
    [-0.35, 4.7, 2.0],  # 'left_eye',        # 2
    [0.35, 4.7, 2.0],  # 'right_eye',       # 3
    [-0.7, 4.5, 2.0],  # 'left_ear',        # 4
    [0.7, 4.5, 2.0],  # 'right_ear',       # 5
    [1.4, 7.0, 2.0],  # 'throat',            # 6
    [7, 2.0, 2.0],  # 'tail',                   # 7
    [2, 4.0, 2.0],  # 'withers',         # 8
    [3.5, 3.0, 2.0],  # 'L_F_elbow',      # 9
    [3, 3.2, 2.0],  # 'R_F_elbow',     # 10
    [6.5, 3.1, 2.0],  # 'L_B_elbow',      # 11
    [6, 3.3, 2.0],  # 'R_B_elbow',     # 12
    [3.5, 3.0, 2.0],  # 'left_elbow',      # 8
    [3, 3.2, 2.0],  # 'right_elbow',     # 9
    [3.5, 1.0, 2.0],  # 'L_F_Knee',     # 13
    [3, 1.2, 2.0],  # 'R_F_Knee',     # 14
    [6.5, 1.1, 2.0],  # 'L_B_Knee',     # 15
    [6, 1.3, 2.0],  # 'R_B_Knee',     # 16
    [3.5, -1.0, 2.0],  # 'L_F_Paw',      # 17
    [3, -1.2, 2.0],  # 'R_F_Paw',      # 18
    [6.5, -1.1, 2.0],  # 'L_B_Paw',      # 19
    [6, -1.3, 2.0],  # 'R_B_Paw',      # 20
])
