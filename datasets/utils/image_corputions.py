import ctypes
from io import BytesIO

import cv2
import numpy as np
import skimage as sk
import torch
from accelerate.commands.menu.input import mark
from PIL import Image as PILImage
from scipy.ndimage import map_coordinates
from scipy.ndimage import zoom as scizoom
from skimage.filters import gaussian
from wand.api import library as wandlibrary
from wand.image import Image as WandImage

# Setup for Wand's motion blur
wandlibrary.MagickMotionBlurImage.argtypes = (
    ctypes.c_void_p,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
)


def plasma_fractal(mapsize=32, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert mapsize & (mapsize - 1) == 0
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
        calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[
            stepsize // 2 : mapsize : stepsize, stepsize // 2 : mapsize : stepsize
        ] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
        calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[
            stepsize // 2 : mapsize : stepsize, stepsize // 2 : mapsize : stepsize
        ]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2 : mapsize : stepsize] = wibbledmean(
            ltsum
        )
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2 : mapsize : stepsize, 0:mapsize:stepsize] = wibbledmean(
            ttsum
        )

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


# === Helpers ===
def disk(radius, alias_blur=0.1, dtype=np.float32):
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X**2 + Y**2) <= radius**2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)
    return cv2.GaussianBlur(aliased_disk, ksize=(3, 3), sigmaX=alias_blur)


def clipped_zoom(img, zoom_factor):
    h, w = img.shape[:2]
    ch = int(np.ceil(h / zoom_factor))
    cw = int(np.ceil(w / zoom_factor))
    top = (h - ch) // 2
    left = (w - cw) // 2
    img = scizoom(
        img[top : top + ch, left : left + cw], (zoom_factor, zoom_factor, 1), order=1
    )
    trim_top = (img.shape[0] - h) // 2
    trim_left = (img.shape[1] - w) // 2
    return img[trim_top : trim_top + h, trim_left : trim_left + w]


def dtype_helper(x: torch.Tensor, marker=None):
    if marker is None:
        if x.dtype == np.uint8:
            return np.array(x) / 255.0, np.uint8
        else:
            return x, x.dtype
    elif marker == np.uint8:
        return (np.clip(x, 0, 1) * 255).astype(marker)
    else:
        return np.clip(x, 0.0, 1.0)


# === Corruption Functions ===


def test(x, severity=1):
    x = np.ones_like(x, dtype=np.uint8)
    return x


def gaussian_noise(x, severity=1):
    c = [0.04, 0.06, 0.08, 0.09, 0.10][severity - 1]
    x, marker = dtype_helper(x)
    x += np.random.normal(size=x.shape, scale=c)
    return dtype_helper(x, marker)


def shot_noise(x, severity=1):
    c = [500, 250, 100, 75, 50][severity - 1]
    x, marker = dtype_helper(x)
    return dtype_helper(np.random.poisson(x * c) / c, marker)


def impulse_noise(x, severity=1):
    c = [0.01, 0.02, 0.03, 0.05, 0.07][severity - 1]
    x, marker = dtype_helper(x)
    return dtype_helper(sk.util.random_noise(x, mode="s&p", amount=c), marker)


def speckle_noise(x, severity=1):
    c = [0.06, 0.1, 0.12, 0.16, 0.2][severity - 1]
    x, marker = dtype_helper(x)

    return dtype_helper(x + x * np.random.normal(size=x.shape, scale=c), marker)


def gaussian_blur(x, severity=1):
    c = [0.4, 0.6, 0.7, 0.8, 1][severity - 1]
    x, marker = dtype_helper(x)
    return dtype_helper(gaussian(x, sigma=c, channel_axis=1), marker)


def defocus_blur(x, severity=1):
    c = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (1, 0.2), (1.5, 0.1)][severity - 1]
    x, marker = dtype_helper(x)
    radius = int(c[0] * x.shape[0])  # scale by image size
    kernel = disk(radius=radius, alias_blur=c[1])
    channels = [cv2.filter2D(x[:, :, d], -1, kernel) for d in range(3)]
    return dtype_helper(np.stack(channels, axis=-1), marker)


def motion_blur(x, severity=1):
    c = [(6, 1), (6, 1.5), (6, 2), (8, 2), (9, 2.5)][severity - 1]
    was_float = False
    if x.dtype == torch.float32:
        x = x * 255
        x.to(torch.uint8)
        was_float = True

    # numpy array → PNG bytes directly
    x_bgr = x[..., [2, 1, 0]]  # RGB → BGR for cv2
    _, encoded = cv2.imencode(".png", x_bgr)

    x = MotionImage(blob=encoded.tobytes())
    x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

    x = cv2.imdecode(np.frombuffer(x.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)
    if x.ndim == 2:
        x = np.stack([x] * 3, axis=-1)
    if was_float:
        return np.clip(x[..., [2, 1, 0]] / 255, 0, 1)
    return np.clip(x[..., [2, 1, 0]], 0, 255)


def jpeg_compression(x, severity=1):
    was_float = False
    if x.dtype == torch.float32:
        was_float = True
    c = [80, 65, 58, 50, 40][severity - 1]
    _, encoded = cv2.imencode(".jpg", x[..., [2, 1, 0]], [cv2.IMWRITE_JPEG_QUALITY, c])
    if was_float:
        return np.clip(
            cv2.imdecode(encoded, cv2.IMREAD_COLOR)[..., [2, 1, 0]] / 255, 1, 0
        )
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)[..., [2, 1, 0]]


def pixelate(x, severity=1):
    c = [0.95, 0.9, 0.85, 0.75, 0.65][severity - 1]
    H, W = x.shape[:2]
    small = cv2.resize(x, (int(W * c), int(H * c)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)


def frost(x, severity=1):
    c = [(1, 0.2), (1, 0.3), (0.9, 0.4), (0.85, 0.4), (0.75, 0.45)][severity - 1]
    H, W = x.shape[:2]
    idx = np.random.randint(3)
    filename = [f"./datasets/utils/frost{i}.jpg" for i in range(4, 7)][idx]
    frost_img = cv2.imread(filename)
    if frost_img is None:
        raise FileNotFoundError(f"Failed to load image at: {filename}")
    orig_h, orig_w = frost_img.shape[:2]
    scale_factor = max(H / orig_h, W / orig_w, 1.0)
    frost_img = cv2.resize(frost_img, (0, 0), fx=scale_factor, fy=scale_factor)
    fh, fw = frost_img.shape[:2]
    y0 = np.random.randint(0, fh - H + 1)
    x0 = np.random.randint(0, fw - W + 1)
    frost_crop = frost_img[y0 : y0 + H, x0 : x0 + W][..., ::-1] / 255.0  # BGR→RGB
    x_arr, marker = dtype_helper(x)
    return dtype_helper(c[0] * x_arr + c[1] * frost_crop, marker)


def snow(x, severity=1):
    c = [
        (0.1, 0.2, 1, 0.6, 8, 3, 0.95),
        (0.1, 0.2, 1, 0.5, 10, 4, 0.9),
        (0.15, 0.3, 1.75, 0.55, 10, 4, 0.9),
        (0.25, 0.3, 2.25, 0.6, 12, 6, 0.85),
        (0.3, 0.3, 1.25, 0.65, 14, 12, 0.8),
    ][severity - 1]
    x_arr, marker = dtype_helper(x)
    H, W = x_arr.shape[:2]
    snow_layer = np.random.normal(loc=c[0], scale=c[1], size=(H, W))[..., None]
    snow_layer = clipped_zoom(snow_layer, c[2])
    snow_layer[snow_layer < c[3]] = 0
    # encode snow_layer directly with cv2 instead of PIL+BytesIO
    snow_uint8 = (snow_layer.squeeze() * 255).astype(np.uint8)
    _, encoded = cv2.imencode(".png", snow_uint8)
    snow_wand = MotionImage(blob=encoded.tobytes())
    snow_wand.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))
    arr = (
        cv2.imdecode(
            np.frombuffer(snow_wand.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED
        )
        / 255.0
    )
    arr = arr[..., None]
    gray = cv2.cvtColor(x_arr, cv2.COLOR_RGB2GRAY)[..., None]
    x_mod = c[6] * x_arr + (1 - c[6]) * np.maximum(x_arr, gray * 1.5 + 0.5)
    return dtype_helper(x_mod + arr + np.rot90(arr, 2), marker)


def fog(x, severity=1):
    c = [(0.2, 3), (0.5, 3), (0.75, 2.5), (1, 2), (1.5, 1.75)][severity - 1]
    x_arr, marker = dtype_helper(x)
    H, W = x_arr.shape[:2]
    max_val = x_arr.max()
    fog_layer = plasma_fractal(mapsize=256, wibbledecay=int(c[1]))
    fog_crop = fog_layer[:H, :W][..., None]
    x_mod = x_arr + c[0] * fog_crop
    return dtype_helper(x_mod * max_val / (max_val + c[0]), marker)


def spatter(x, severity=1):
    c = [
        (0.62, 0.1, 0.7, 0.7, 0.5, 0),
        (0.65, 0.1, 0.8, 0.7, 0.5, 0),
        (0.65, 0.3, 1, 0.69, 0.5, 0),
        (0.65, 0.1, 0.7, 0.69, 0.6, 1),
        (0.65, 0.1, 0.5, 0.68, 0.6, 1),
    ][severity - 1]

    x, marker = dtype_helper(x)
    H, W = x.shape[:2]
    liquid = np.random.normal(loc=c[0], scale=c[1], size=(H, W))

    liquid = gaussian(liquid, sigma=c[2])
    liquid[liquid < c[3]] = 0

    if c[5] == 0:
        liquid_img = (liquid * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_img, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        dist = np.clip(dist, 0, 20)
        dist = cv2.blur(dist.astype(np.uint8), (3, 3))
        dist = cv2.equalizeHist(dist)

        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor((liquid * dist).astype(np.float32), cv2.COLOR_GRAY2BGR)
        m = np.clip(m / m.max() * c[4], 0, 1)

        color = np.stack(
            [
                np.full_like(m[..., 0], 175 / 255.0),
                np.full_like(m[..., 1], 238 / 255.0),
                np.full_like(m[..., 2], 238 / 255.0),
            ],
            axis=-1,
        )

        return dtype_helper(x + m * color, marker)

    else:
        m = (liquid > c[3]).astype(np.float32)
        m = gaussian(m, sigma=c[4])
        m[m < 0.8] = 0

        mud = np.stack(
            [
                np.full_like(x[..., 0], 63 / 255.0),
                np.full_like(x[..., 1], 42 / 255.0),
                np.full_like(x[..., 2], 20 / 255.0),
            ],
            axis=-1,
        )

        color = mud * m[..., None]
        x *= 1 - m[..., None]
        return dtype_helper(x + m * color, marker)


def contrast(x, severity=1):
    c = [0.75, 0.5, 0.4, 0.3, 0.15][severity - 1]
    x, marker = dtype_helper(x)
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return dtype_helper((x - means) * c + means, marker)


def brightness(x, severity=1):
    c = [0.05, 0.1, 0.15, 0.2, 0.3][severity - 1]
    x, marker = dtype_helper(x)
    hsv = sk.color.rgb2hsv(x)
    hsv[..., 2] = np.clip(hsv[..., 2] + c, 0, 1)
    return dtype_helper(sk.color.hsv2rgb(hsv), marker)


def saturate(x, severity=1):
    c = [(0.3, 0), (0.1, 0), (1.5, 0), (2, 0.1), (2.5, 0.2)][severity - 1]
    x, marker = dtype_helper(x)
    hsv = sk.color.rgb2hsv(x)
    hsv[..., 1] = np.clip(hsv[..., 1] * c[0] + c[1], 0, 1)
    return dtype_helper(sk.color.hsv2rgb(hsv), marker)


def elastic_transform(image, severity=1):
    IMSIZE = image.size  # handles 32 or 64 automatically
    c = [
        (IMSIZE * 0, IMSIZE * 0, IMSIZE * 0.08),
        (IMSIZE * 0.05, IMSIZE * 0.2, IMSIZE * 0.07),
        (IMSIZE * 0.08, IMSIZE * 0.06, IMSIZE * 0.06),
        (IMSIZE * 0.1, IMSIZE * 0.04, IMSIZE * 0.05),
        (IMSIZE * 0.1, IMSIZE * 0.03, IMSIZE * 0.03),
    ][severity - 1]

    image, marker = dtype_helper(image)

    shape = image.shape
    shape_size = shape[:2]

    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32(
        [
            center_square + square_size,
            [center_square[0] + square_size, center_square[1] - square_size],
            center_square - square_size,
        ]
    )
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)

    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(
        image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101
    )

    dx = (
        gaussian(np.random.uniform(-1, 1, size=shape[:2]), c[1], mode="reflect") * c[0]
    ).astype(np.float32)
    dy = (
        gaussian(np.random.uniform(-1, 1, size=shape[:2]), c[1], mode="reflect") * c[0]
    ).astype(np.float32)
    dx, dy = dx[..., None], dy[..., None]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = (
        np.reshape(y + dy, (-1, 1)),
        np.reshape(x + dx, (-1, 1)),
        np.reshape(z, (-1, 1)),
    )

    return dtype_helper(
        map_coordinates(image, indices, order=1, mode="reflect").reshape(shape), marker
    )


def glass_blur(x, severity=1):
    # (sigma, max_delta, iterations)
    c = [(0.05, 1, 1), (0.25, 1, 1), (0.4, 1, 1), (0.25, 1, 2), (0.4, 1, 2)][
        severity - 1
    ]
    sigma, delta, iterations = c

    x, marker = dtype_helper(x)
    x = np.clip(gaussian(x, sigma=sigma, channel_axis=-1), 0, 1)
    x = np.uint8(x * 255)

    H, W = x.shape[:2]

    for _ in range(iterations):
        for h in range(delta, H - delta):
            for w in range(delta, W - delta):
                dx, dy = np.random.randint(-delta, delta + 1, size=2)
                h_new, w_new = h + dy, w + dx
                if 0 <= h_new < H and 0 <= w_new < W:
                    tmp = x[h, w].copy()
                    x[h, w] = x[h_new, w_new]
                    x[h_new, w_new] = tmp

    x = x / 255.0
    return dtype_helper(gaussian(x, sigma=sigma, channel_axis=-1), marker)


def zoom_blur(x, severity=1):
    c = [
        np.arange(1, 1.06, 0.01),
        np.arange(1, 1.11, 0.01),
        np.arange(1, 1.16, 0.01),
        np.arange(1, 1.21, 0.01),
        np.arange(1, 1.26, 0.01),
    ][severity - 1]

    x, marker = dtype_helper(x)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return dtype_helper(x, marker)


def clean(x, severity=1):
    return x


# === Exported Dictionary ===
corruption_dict = {
    "test": test,  # just for testing purposes, not to be used
    "gaussian_noise": gaussian_noise,
    "shot_noise": shot_noise,
    "impulse_noise": impulse_noise,
    "defocus_blur": defocus_blur,
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "speckle_noise": speckle_noise,
    "jpeg_compression": jpeg_compression,
    "pixelate": pixelate,
    "frost": frost,
    "snow": snow,
    "fog": fog,
    "spatter": spatter,
    "contrast": contrast,
    "brightness": brightness,
    "saturate": saturate,
    "elastic_transform": elastic_transform,
    "glass_blur": glass_blur,
    "zoom_blur": zoom_blur,
    "clean": clean,
}
