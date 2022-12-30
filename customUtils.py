def findMinMax(landmarks, width, height):
    x_max, y_max, z_max = 0, 0, 0
    x_min, y_min, z_min = width, height, None

    for landmark in landmarks:
        x, y, z = landmark.x, landmark.y, landmark.z
        if x > x_max:
            x_max = x
        if x < x_min:
            x_min = x
        if y > y_max:
            y_max = y
        if y < y_min:
            y_min = y
        if z_min == None or z < z_min:
            z_min = z
        if z > z_max:
            z_max = z
    return x_min, y_min, z_min, x_max, y_max, z_max
