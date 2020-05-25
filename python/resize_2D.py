# https://machinethink.net/blog/coreml-upsampling/
import numpy as np
import matplotlib.pyplot as plt


def resize_bilinear(input_img, scale_factor_h, scale_factor_w):
    source_h = input_img.shape[0]
    source_w = input_img.shape[1]

    resized_h = int(source_h * scale_factor_h)
    resized_w = int(source_w * scale_factor_w)

    output = np.zeros((resized_h, resized_w), dtype=np.float32)

    def read_pixel(x, y):
        x = np.clip(x, 0, source_w - 1)
        y = np.clip(y, 0, source_h - 1)
        return input_img[y, x]

    def bilinear_interpolate(x, y):
        x1 = int(np.floor(x))
        x2 = x1 + 1

        y1 = int(np.floor(y))
        y2 = y1 + 1

        P11 = read_pixel(x1, y1)
        P12 = read_pixel(x1, y2)
        P21 = read_pixel(x2, y1)
        P22 = read_pixel(x2, y2)

        return (P11 * (x2 - x) * (y2 - y) +
                P12 * (x2 - x) * (y - y1) +
                P21 * (x - x1) * (y2 - y) +
                P22 * (x - x1) * (y - y1)) / ((x2 - x1) * (y2 - y1))

    for dst_y in range(resized_h):
        for dst_x in range(resized_w):
            src_x = (dst_x + 0.5) / scale_factor_w - 0.5
            src_y = (dst_y + 0.5) / scale_factor_h - 0.5
            output[dst_y, dst_x] = bilinear_interpolate(src_x, src_y)

    return output

np.random.seed(12345)
feature_map = np.random.randn(9, 10).astype(np.float32)
plt.figure()
plt.imshow(feature_map)


output = resize_bilinear(feature_map, 2, 2)
plt.figure()
plt.imshow(output)
plt.show()
