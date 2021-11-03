color_mapping_1 = [
    (128, 0, 0),
    (170, 0, 0),
    (213, 0, 0),
    (255, 0, 0),
    (255, 43, 0),
    (255, 85, 0),
    (255, 128, 0),
    (255, 170, 0),
    (255, 213, 0),
    (255, 255, 0),
    (213, 255, 85),
    (213, 255, 43),
    (170, 255, 85),
    (128, 255, 128),
    (85, 255, 170),
    (43, 255, 213),
    (0, 255, 255),
    (0, 213, 255),
    (0, 170, 255),
    (0, 128, 255),
    (0, 85, 255),
    (0, 43, 255),
    (0, 0, 255),
    (0, 0, 213),
    (0, 0, 170),
]


def filter_colors(all_vertebraes):
    result = []
    step = len(color_mapping_1) // len(all_vertebraes) // 2
    for i,v in enumerate(all_vertebraes, start=0):
        result.append(color_mapping_1[step*i])
    return result
