import argparse
import numpy as np
import cv2

def compute_fft_magnitude(gray_image):
    f = np.fft.fft2(gray_image.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    mag = 1 - np.log1p(mag)  # log(1 + |F|)
    # normalize to [0, 1]
    mn, mx = float(mag.min()), float(mag.max())
    if mx - mn < 1e-8:
        return np.zeros_like(mag, dtype=np.float32)
    mag = (mag - mn) / (mx - mn)
    return mag

def compute_gradient_magnitude(gray_image):
    gradient_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)
    return gradient_magnitude

def smooth_1d(v, k = 17):
    """Simple 1D smoothing with a Gaussian-like kernel (no scipy)."""
    k = int(k)
    if k < 3:
        return v
    if k % 2 == 0:
        k += 1
    sigma = k / 6.0
    x = np.arange(k) - k // 2
    ker = np.exp(-(x * x) / (2 * sigma * sigma))
    ker = ker / (ker.sum() + 1e-8)
    vv = np.convolve(v, ker, mode="same")
    return vv

def detect_peak(proj, rel_thr=0.35, min_dist=6):
    print("Detecting peaks...")
    center = len(proj) // 2

    mx = float(proj.max())
    if mx < 1e-6:
        return None

    thr = mx * float(rel_thr)

    peak_width = 8  # to enforce local max over this width
    
    candidates = []
    for i in range(1, len(proj) - 1):
        is_peak = True
        for j in range(1, peak_width):
            if i - j < 0 or i + j >= len(proj):
                continue
            if proj[i-j+1] < proj[i - j] or proj[i+j-1] < proj[i + j]:
                is_peak = False
                break  
        if is_peak and proj[i] >= thr:
            left_climb = 0
            for k in range(i, 0, -1):
                if proj[k] > proj[k-1]:
                    left_climb = abs(proj[i] - proj[k-1])
                else:
                    break

            right_fall = 0
            for k in range(i, len(proj) - 1):
                if proj[k] > proj[k+1]:
                    right_fall = abs(proj[i] - proj[k+1])
                else:
                    break
            
            candidates.append({
                'index': i,
                'climb': left_climb,
                'fall': right_fall,
                'score': max(left_climb, right_fall)
            })

    if not candidates:
        print("No peaks found.")
        return None

    # enforce a dead-zone around center
    left = [i for i in candidates if i['index'] < center - min_dist and i['index'] > center * 0.25]
    right = [i for i in candidates if i['index'] > center + min_dist and i['index'] < center * 1.75]

    left.sort(key=lambda x: x['score'], reverse=True)
    right.sort(key=lambda x: x['score'], reverse=True)

    if not left or not right:
        print("Not enough peaks found on both sides.")
        return None

    # pick nearest to center on each side
    peak_left = left[0]['index']   
    peak_right = right[0]['index']  
    print(f"Detected peaks at: left = {peak_left}, right = {peak_right}")

    return abs(peak_right - peak_left)/2

def resize_by_center_sampling(image, out_w, out_h, offset, random_offset = False):
    H, W = image.shape[:2]
    pixel_size_x = W / out_w
    pixel_size_y = H / out_h
    offset_x, offset_y = offset
    # use a random bias to avoid sampling artifacts
    if random_offset:
        offset_x += (np.random.rand() * 0.2 + 0.4) * pixel_size_x
        offset_y += (np.random.rand() * 0.2 + 0.4) * pixel_size_y
    else:
        offset_x += 0.5 * pixel_size_x
        offset_y += 0.5 * pixel_size_y

    xs = np.arange(out_w) * pixel_size_x + offset_x
    ys = np.arange(out_h) * pixel_size_y + offset_y

    xi = np.rint(xs).astype(np.int32)
    yi = np.rint(ys).astype(np.int32)

    xi = np.clip(xi, 0, W - 1)
    yi = np.clip(yi, 0, H - 1)

    if image.ndim == 2:
        return image[yi[:, None], xi[None, :]]
    else:
        return image[yi[:, None], xi[None, :], :]

def find_best_grid(origin, range_val_min, range_val_max, grad_mag, thr = 0):
    best = origin
    peaks = []
    mx = np.max(grad_mag)
    if mx < 1e-6:
        return best
    rel_thr = mx * thr
    for i in range(-int(range_val_min + 0.5), int(range_val_max + 0.5)+1):
        candidate = int(origin + i)
        if candidate <= 0 or candidate >= len(grad_mag):
            continue
        if grad_mag[candidate] > grad_mag[candidate -1] and grad_mag[candidate] > grad_mag[candidate +1] and grad_mag[candidate] >= rel_thr:
            peaks.append((grad_mag[candidate], candidate))
    if len(peaks) == 0:
        return best
    
    # find the brightest peak
    peaks.sort(key=lambda x: x[0], reverse=True)
    best = peaks[0][1]
    return best

def sample_center(image, x_coords, y_coords):
    x = np.asarray(x_coords)
    y = np.asarray(y_coords)

    centers_x = ((x[1:] + x[:-1]) * 0.5).astype(np.int32)
    centers_y = ((y[1:] + y[:-1]) * 0.5).astype(np.int32)

    scaled_image = image[centers_y[:, None], centers_x[None, :]]
    return scaled_image

def sample_majority(image, x_coords, y_coords, max_samples=256, iters=6, seed=0):
    rng = np.random.default_rng(seed)

    img = image.astype(np.float32) if image.dtype != np.float32 else image
    H, W = img.shape[:2]
    if img.ndim == 2:
        img = img[..., None]
    C = img.shape[2]

    x = np.asarray(x_coords, dtype=np.int32)
    y = np.asarray(y_coords, dtype=np.int32)

    nx, ny = len(x) - 1, len(y) - 1
    out = np.empty((ny, nx, C), dtype=np.float32)

    for j in range(ny):
        y0, y1 = int(y[j]), int(y[j + 1])
        y0 = np.clip(y0, 0, H); y1 = np.clip(y1, 0, H)
        if y1 <= y0: y1 = min(y0 + 1, H)

        for i in range(nx):
            x0, x1 = int(x[i]), int(x[i + 1])
            x0 = np.clip(x0, 0, W); x1 = np.clip(x1, 0, W)
            if x1 <= x0: x1 = min(x0 + 1, W)

            cell = img[y0:y1, x0:x1].reshape(-1, C)
            n = cell.shape[0]
            if n == 0:
                out[j, i] = 0
                continue
            if n > max_samples:
                cell = cell[rng.integers(0, n, size=max_samples)]

            # k=2 init：取第一个点 + 最远点（稳定）
            c0 = cell[0]
            c1 = cell[np.argmax(((cell - c0) ** 2).sum(1))]

            for _ in range(iters):
                d0 = ((cell - c0) ** 2).sum(1)
                d1 = ((cell - c1) ** 2).sum(1)
                m1 = d1 < d0

                if np.any(~m1): c0 = cell[~m1].mean(0)
                if np.any(m1):  c1 = cell[m1].mean(0)

            out[j, i] = c1 if m1.sum() >= (~m1).sum() else c0

    if image.dtype == np.uint8:
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return out

def refine_grids(image, grid_x, grid_y):
    H, W = image.shape[:2]
    x_coords = []
    y_coords = []
    cell_w = W / grid_x
    cell_h = H / grid_y

    # calculate gradient magnitude
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)

    grad_x_sum = np.sum(np.abs(grad_x), axis=0).reshape(-1)
    grad_y_sum = np.sum(np.abs(grad_y), axis=1).reshape(-1)

    # refine grid lines based on gradient magnitude from center
    x = find_best_grid(W / 2, cell_w, cell_w, grad_x_sum)
    while(x < W):
        x = find_best_grid(x, cell_w / 4, cell_w / 4, grad_x_sum)
        x_coords.append(x)
        x += cell_w
    x = find_best_grid(W / 2, cell_w, cell_w, grad_x_sum) - cell_w
    while(x > 0):
        x = find_best_grid(x, cell_w / 4, cell_w / 4, grad_x_sum)
        x_coords.append(x)
        x -= cell_w

    y = find_best_grid(H / 2, cell_h, cell_h, grad_y_sum)
    while(y < H):
        y = find_best_grid(y, cell_h / 4, cell_h / 4, grad_y_sum)   
        y_coords.append(y)
        y += cell_h
    y = find_best_grid(H / 2, cell_h, cell_h, grad_y_sum) - cell_h
    while(y > 0):
        y = find_best_grid(y, cell_h / 4, cell_h / 4, grad_y_sum)   
        y_coords.append(y)
        y -= cell_h

    x_coords = sorted(x_coords)
    y_coords = sorted(y_coords)
    
    return x_coords, y_coords

def detect_pixel_size(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    # Compute FFT
    mag = compute_fft_magnitude(gray)

    # Compute projections
    band_row = W // 2
    band_col = H // 2

    row_sum = np.sum(mag[:, W//2 - band_row: W//2 + band_row], axis=1)
    col_sum = np.sum(mag[H//2 - band_col: H//2 + band_col, :], axis=0)

    row_sum = cv2.normalize(row_sum, None, 0, 1, cv2.NORM_MINMAX).flatten()
    col_sum = cv2.normalize(col_sum, None, 0, 1, cv2.NORM_MINMAX).flatten()

    # Smooth projections
    row_sum = smooth_1d(row_sum, k=17)
    col_sum = smooth_1d(col_sum, k=17)

    # Detect peaks
    scale_row = detect_peak(row_sum)
    scale_col = detect_peak(col_sum)
    if scale_row is None or scale_col is None:
        print("Cannot detect pixel size.")
        return None, None

    pixel_size_x = W / scale_col
    pixel_size_y = H / scale_row

    return pixel_size_x, pixel_size_y

def grid_layout(image, x_coords, y_coords):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(image)
    plt.title("Scaled Image by Grid Sampling")
    for x in x_coords:
        plt.axvline(x=x, linewidth=0.6)
    for y in y_coords:
        plt.axhline(y=y, linewidth=0.6)
    plt.show()

def get_perfect_pixel(image, sample_method="center", debug=False):

    H, W = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute FFT
    mag = compute_fft_magnitude(gray)

    # Compute projections
    band_row = W // 2
    band_col = H // 2

    row_sum = np.sum(mag[:, W//2 - band_row: W//2 + band_row], axis=1)
    col_sum = np.sum(mag[H//2 - band_col: H//2 + band_col, :], axis=0)

    row_sum = cv2.normalize(row_sum, None, 0, 1, cv2.NORM_MINMAX).flatten()
    col_sum = cv2.normalize(col_sum, None, 0, 1, cv2.NORM_MINMAX).flatten()

    # Smooth projections
    row_sum = smooth_1d(row_sum, k=17)
    col_sum = smooth_1d(col_sum, k=17)

    # Detect peaks
    scale_row = detect_peak(row_sum)
    scale_col = detect_peak(col_sum)

    if(scale_col == 0):
        print("No pixel detected")
        return None, None, image

    pixel_size_x = W / scale_col
    pixel_size_y = H / scale_row
    pixel_size = min(pixel_size_x, pixel_size_y)
    if(pixel_size <= 3.0):
        print("Detected pixel size is too small.")
        return None, None, image
    print("Estimated pixel size: {:.2f} pixels".format(pixel_size))

    size_x = int(round(scale_col))
    size_y = int(round(scale_row))

    # scaled_image = resize_by_center_sampling(image, size_x, size_y, best_offset)

    # refine grid lines
    x_coords, y_coords = refine_grids(image, size_x, size_y)
    refined_size_x = len(x_coords) - 1
    refined_size_y = len(y_coords) - 1
    print(f"Refined grid size: ({refined_size_x}, {refined_size_y})")

    # sample by majority
    if sample_method == "majority":
        scaled_image = sample_majority(image, x_coords, y_coords)

    # sample by center
    else:
        scaled_image = sample_center(image, x_coords, y_coords)

    # debug
    if debug:
        grid_layout(image, x_coords, y_coords)

    return refined_size_x, refined_size_y, scaled_image
