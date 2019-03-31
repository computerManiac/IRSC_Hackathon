import numpy as np

def rect2bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	return (x,y,w,h)

def shape2arr(shape):
	coords = np.zeros((68,2), dtype=int)

	for i in range(0,68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	return coords