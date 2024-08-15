# %%
import cv2
import numpy as np

# %%
size = (60, 45)
img = cv2.imread('map.png')
img = cv2.resize(img, size)

CR = {
    "r": [np.array([0, 0, 128]),
          np.array([128, 128, 255])],
    "y": [np.array([0, 128, 128]),
          np.array([128, 255, 255])],
    "g": [np.array([0, 128, 0]),
          np.array([128, 255, 128])],
    "b": [np.array([128, 0, 0]),
          np.array([255, 128, 128])],
    "c": [np.array([128, 128, 0]),
          np.array([255, 255, 128])],
    "m": [np.array([128, 0, 128]),
          np.array([255, 128, 255])],
}

# %%
# obstacles=1
map = cv2.inRange(img, CR['m'][0], CR['m'][1]).astype(bool).astype(int)
# building=2
map += cv2.inRange(img, CR['b'][0], CR['b'][1]).astype(bool).astype(int) * 2

# %%
# deployment point=3
deployment = cv2.inRange(img, CR['b'][0], CR['b'][1])

contours, _ = cv2.findContours(deployment, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    ((x, y), (h, w), _) = cv2.minAreaRect(contour)
    map[int(np.round(y)), int(np.round(x))] = 3
    map[int(np.round(y - h / 3)), int(np.round(x - w / 3))] = 3
    map[int(np.round(y + h / 3)), int(np.round(x - w / 3))] = 3
    map[int(np.round(y - h / 3)), int(np.round(x + w / 3))] = 3
    map[int(np.round(y + h / 3)), int(np.round(x + w / 3))] = 3
# %%
# entrances=4
entrances = cv2.inRange(img, CR['g'][0], CR['g'][1])

contours, _ = cv2.findContours(entrances, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    ((x, y), _) = cv2.minEnclosingCircle(contour)
    map[int(np.ceil(y)), int(np.ceil(x))] = 4
# %%
# exit=5
exit = cv2.inRange(img, CR['c'][0], CR['c'][1])

contours, _ = cv2.findContours(exit, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    ((x, y), _) = cv2.minEnclosingCircle(contour)
    map[int(np.ceil(y)), int(np.ceil(x))] = 5
# %%
# waypoint=6
waypoint = cv2.inRange(img, CR['y'][0], CR['y'][1])

contours, _ = cv2.findContours(waypoint, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    ((x, y), _) = cv2.minEnclosingCircle(contour)
    map[int(np.ceil(y)), int(np.ceil(x))] = 6

# %%
map = map.tolist()
with open('map.csv', 'w') as f:
    for idx, line in enumerate(map):
        if idx % 2 == 0:
            if idx in [0, len(map) - 1]:
                line.insert(-1, 2)
            else:
                line.insert(-1, 0)
        line = ','.join([str(x) for x in line]) + '\n'
        f.write(line)
# %%
