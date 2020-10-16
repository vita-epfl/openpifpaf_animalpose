

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