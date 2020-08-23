import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance


# INPUT:
# shape- shape of the generated array (tuple of 2 ints) [shape must be a multiple of res]
# res- number of periods of noise to generate along each axis (tuple of 2 ints)
# tileable- if the noise should be tileable along each axis (tuple of 2 bools)

def generate_perlin_noise_2d(shape, res, tileable=(False, False)):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    if tileable[0]:
        gradients[-1,:] = gradients[0,:]
    if tileable[1]:
        gradients[:,-1] = gradients[:,0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[    :-d[0],    :-d[1]]
    g10 = gradients[d[0]:     ,    :-d[1]]
    g01 = gradients[    :-d[0],d[1]:     ]
    g11 = gradients[d[0]:     ,d[1]:     ]
    # Ramps
    n00 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]  )) * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]  )) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)

# INPUT:
# shape: shape of the generated array (tuple of 2 ints)
# res: number of periods of noise to generate along each axis (tuple of 2 ints)
# octaves: number of octaves in the noise (int)
# persistence: scaling factor between two octaves (float)
# lacunarity: frequency factor between two octaves (float)
# tileable: if the noise should be tileable along each axis (tuple of 2 bools)

def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5, lacunarity=1, tileable=(False, False)):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency*res[0], frequency*res[1]), tileable)
        frequency *= lacunarity
        amplitude *= persistence
    return noise

def add_noise(im):
    #np.random.seed(0)
    y_axis, x_axis ,_ = im.shape
    scale = (y_axis, x_axis)
    res = 6
    noise = generate_fractal_noise_2d(scale, (res, res), )
    im=im.astype(np.float64)
    im /=255.
    b, g, r = cv.split(im)
    #  adding noise to image as weighted average
    b = 0.6*b+0.4*noise
    g = 0.6*g+0.4*noise
    r = 0.6*r+0.4*noise
    noisy_im = np.dstack((r,g,b))
    return noisy_im
