import numpy as np
import scipy.signal 
import cv2
import time 
import matplotlib.pyplot as plt
import math 
import skimage.transform as sktr 
import skimage.io as skio 
import os

# question 1.1: 
def pad_image(img: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

def conv2d_4loops(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    H, W = img.shape
    kH, kW = kernel.shape
    kernel_flipped = kernel[::-1, ::-1]  # flip filter horizontally & vertically
    
    padded_img = pad_image(img, kH // 2, kW // 2)
    out = np.zeros((H, W), dtype=np.float64)
    
    for i in range(H):
        for j in range(W):
            val = 0.0
            for m in range(kH):
                for n in range(kW):
                    val += padded_img[i + m, j + n] * kernel_flipped[m, n]
            out[i, j] = val
    return out

def conv2d_2loops(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    H, W = img.shape
    kH, kW = kernel.shape
    kernel_flipped = kernel[::-1, ::-1]
    
    padded_img = pad_image(img, kH // 2, kW // 2)
    out = np.zeros((H, W), dtype=np.float64)
    
    for i in range(H):
        for j in range(W):
            patch = padded_img[i:i + kH, j:j + kW]
            out[i, j] = np.sum(patch * kernel_flipped)
    return out

def question_1_1(): 
    img_path = 'self_portrait.jpg'  
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    img = img.astype(np.float64) / 255.0  # Normalize to [0, 1]

    # filters: 
    box_9x9 = np.ones((9, 9), dtype=np.float64) / 81.0
    Dx = np.array([[1, -1]], dtype=np.float64)
    Dy = np.array([[1], [-1]], dtype=np.float64)

    t0 = time.time()
    res_4loop = conv2d_4loops(img, box_9x9)
    t_4loop = time.time() - t0
    print(f"4-Loop Convolution Time: {t_4loop:.4f} seconds")

    t0 = time.time()
    res_2loop = conv2d_2loops(img, box_9x9)
    t_2loop = time.time() - t0
    print(f"2-Loop Convolution Time: {t_2loop:.4f} seconds")

    t0 = time.time()
    res_scipy = scipy.signal.convolve2d(img, box_9x9, mode='same', boundary='fill', fillvalue=0)
    t_scipy = time.time() - t0
    print(f"SciPy convolve2d Time: {t_scipy:.4f} seconds")


    diff_2loop = np.max(np.abs(res_2loop - res_scipy))
    diff_4loop = np.max(np.abs(res_4loop - res_scipy))
    print(f"Max difference (2-Loop vs SciPy): {diff_2loop:.2e}")
    print(f"Max difference (4-Loop vs SciPy): {diff_4loop:.2e}")

    # apply derivative filters 
    res_dx = conv2d_2loops(img, Dx)
    res_dy = conv2d_2loops(img, Dy)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(img, cmap='gray')
    axes[0].axis('off')

    axes[1].imshow(res_2loop, cmap='gray')
    axes[1].axis('off')

    axes[2].imshow(res_dx, cmap='gray')
    axes[2].axis('off')

    axes[3].imshow(res_dy, cmap='gray')
    axes[3].axis('off')

    plt.tight_layout()

    plt.savefig("part1_1_results.png", dpi=300, bbox_inches='tight')
    plt.show()

# question 1.2: 
def question_1_2(): 
    # Compute partial derivatives on Cameraman image
    cameraman = cv2.imread('cameraman.png', cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0

    Dx = np.array([[1, -1]], dtype=np.float64)
    Dy = np.array([[1], [-1]], dtype=np.float64)

    # Partial derivatives
    grad_x = scipy.signal.convolve2d(cameraman, Dx, mode='same', boundary='symm')
    grad_y = scipy.signal.convolve2d(cameraman, Dy, mode='same', boundary='symm')

    # Gradient magnitude
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Binarize with empirical threshold
    threshold = 0.25
    edge_binary = (grad_mag > threshold).astype(np.float64)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(grad_x, cmap='gray')
    axes[0].axis('off')

    axes[1].imshow(grad_y, cmap='gray')
    axes[1].axis('off')

    axes[2].imshow(grad_mag, cmap='gray')
    axes[2].axis('off')

    axes[3].imshow(edge_binary, cmap='gray')
    axes[3].axis('off')

    plt.tight_layout()

    plt.savefig("part1_2_results.png", dpi=300, bbox_inches='tight')
    plt.show()

# question 1.3: 
def question_1_3():
    cameraman = cv2.imread('cameraman.png', cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
    Dx = np.array([[1, -1]], dtype=np.float64)
    Dy = np.array([[1], [-1]], dtype=np.float64)
    
    # smooth image first, then get derivatives 
    kernel_1d = cv2.getGaussianKernel(ksize=9, sigma=1.5)
    gaussian_2d = np.outer(kernel_1d, kernel_1d)

    # smooth cameraman image
    cameraman_smoothed = scipy.signal.convolve2d(cameraman, gaussian_2d, mode='same', boundary='symm')

    # derivative of smoothed image
    grad_x_smooth = scipy.signal.convolve2d(cameraman_smoothed, Dx, mode='same', boundary='symm')
    grad_y_smooth = scipy.signal.convolve2d(cameraman_smoothed, Dy, mode='same', boundary='symm')

    grad_mag_smooth = np.sqrt(grad_x_smooth**2 + grad_y_smooth**2)
    edge_binary_smooth = (grad_mag_smooth > 0.08).astype(np.float64)

    # create derivative of gaussian (dog) filters 
    DoG_x = scipy.signal.convolve2d(gaussian_2d, Dx, mode='same')
    DoG_y = scipy.signal.convolve2d(gaussian_2d, Dy, mode='same')

    # single convolution using dog filters
    grad_x_dog = scipy.signal.convolve2d(cameraman, DoG_x, mode='same', boundary='symm')
    grad_y_dog = scipy.signal.convolve2d(cameraman, DoG_y, mode='same', boundary='symm')

    grad_mag_dog = np.sqrt(grad_x_dog**2 + grad_y_dog**2)

    fig, axes = plt.subplots(1, 8, figsize=(18, 5))

    axes[0].imshow(cameraman_smoothed, cmap='gray')
    axes[0].axis('off')

    axes[1].imshow(grad_x_smooth, cmap='gray')
    axes[1].axis('off')

    axes[2].imshow(grad_y_smooth, cmap='gray')
    axes[2].axis('off')

    axes[3].imshow(grad_mag_smooth, cmap='gray')
    axes[3].axis('off')

    axes[4].imshow(edge_binary_smooth, cmap='gray')
    axes[4].axis('off')

    axes[5].imshow(grad_x_dog, cmap='gray')
    axes[5].axis('off')

    axes[6].imshow(grad_y_dog, cmap='gray')
    axes[6].axis('off')

    axes[7].imshow(grad_mag_dog, cmap='gray')
    axes[7].axis('off')

    plt.tight_layout()

    plt.savefig("part1_3_results.png", dpi=300, bbox_inches='tight')
    plt.show()

# question 2.1: 
def unsharp_mask(img: np.ndarray, sigma: float = 2.0, alpha: float = 1.5) -> tuple:
    ksize = int(6 * sigma + 1) | 1
    k1d = cv2.getGaussianKernel(ksize, sigma)
    g2d = np.outer(k1d, k1d)
    
    blurred = np.zeros_like(img)
    if img.ndim == 2:
        blurred = scipy.signal.convolve2d(img, g2d, mode='same', boundary='symm')
    else:
        for c in range(img.shape[2]):
            blurred[:, :, c] = scipy.signal.convolve2d(img[:, :, c], g2d, mode='same', boundary='symm')
            
    high_freq = img - blurred
    sharpened = img + alpha * high_freq
    
    return np.clip(blurred, 0, 1), np.clip(high_freq, -1, 1), np.clip(sharpened, 0, 1)

def question_2_1(): 
    img_bgr = cv2.imread('taj.jpg')
    taj = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

    img_bgr = cv2.imread('duomo.jpg')
    duomo = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

    blurred_taj, high_freq_taj, sharp_taj = unsharp_mask(taj, sigma=2.0, alpha=1.5)
    blurred_duomo, high_freq_duomo, sharp_duomo = unsharp_mask(duomo, sigma=2.0, alpha=1.5)

    # display results 
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(taj); axes[0].set_title("Original")
    axes[1].imshow(blurred_taj); axes[1].set_title(r"Blurred ($\sigma=2$)")
    axes[2].imshow((high_freq_taj + 1) / 2); axes[2].set_title("High Frequencies")
    axes[3].imshow(sharp_taj); axes[3].set_title(r"Sharpened ($\alpha=1.5$)")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("part2_1_taj_unsharp_mask.png", dpi=300)
    plt.show()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(duomo); axes[0].set_title("Original")
    axes[1].imshow(blurred_duomo); axes[1].set_title(r"Blurred ($\sigma=2$)")
    axes[2].imshow((high_freq_duomo + 1) / 2); axes[2].set_title("High Frequencies")
    axes[3].imshow(sharp_duomo); axes[3].set_title(r"Sharpened ($\alpha=1.5$)")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("part2_1_duomo_unsharp_mask.png", dpi=300)
    plt.show()

# question 2.2
def get_points(im1: np.ndarray, im2: np.ndarray) -> tuple:
    print('Please select 2 points in each image for alignment.')
    plt.imshow(im1)
    p1, p2 = plt.ginput(2)
    plt.close()
    plt.imshow(im2)
    p3, p4 = plt.ginput(2)
    plt.close()
    return (p1, p2, p3, p4)

def recenter(im: np.ndarray, r: float, c: float) -> np.ndarray:
    R, C = im.shape[:2]
    rpad = int(np.abs(2*r+1 - R))
    cpad = int(np.abs(2*c+1 - C))
    pad_width = [(0 if r > (R-1)/2 else rpad, 0 if r < (R-1)/2 else rpad),
                 (0 if c > (C-1)/2 else cpad, 0 if c < (C-1)/2 else cpad)]
    if im.ndim == 3:
        pad_width.append((0, 0))
    return np.pad(im, pad_width, 'constant')

def find_centers(p1: tuple, p2: tuple) -> tuple:
    cx = np.round(np.mean([p1[0], p2[0]]))
    cy = np.round(np.mean([p1[1], p2[1]]))
    return cx, cy

def align_image_centers(im1: np.ndarray, im2: np.ndarray, pts: tuple) -> tuple:
    p1, p2, p3, p4 = pts

    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)

    im1 = recenter(im1, cy1, cx1)
    im2 = recenter(im2, cy2, cx2)
    return im1, im2

def rescale_images(im1: np.ndarray, im2: np.ndarray, pts: tuple) -> tuple:
    p1, p2, p3, p4 = pts
    len1 = np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
    len2 = np.sqrt((p4[1] - p3[1])**2 + (p4[0] - p3[0])**2)
    dscale = len2/len1
    channel_axis = -1 if im1.ndim == 3 else None
    if dscale < 1:
        im1 = sktr.rescale(im1, dscale, channel_axis=channel_axis)
    else:
        im2 = sktr.rescale(im2, 1./dscale, channel_axis=channel_axis)
    return im1, im2

def rotate_im1(im1: np.ndarray, pts: tuple) -> tuple:
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), (p2[0] - p1[0]))
    theta2 = math.atan2(-(p4[1] - p3[1]), (p4[0] - p3[0]))
    dtheta = theta2 - theta1
    im1 = sktr.rotate(im1, dtheta*180/np.pi)
    return im1, dtheta

def match_img_size(im1: np.ndarray, im2: np.ndarray) -> tuple:
    # make images the same size
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    if h1 < h2:
        im2 = im2[int(np.floor((h2-h1)/2.)) : -int(np.ceil((h2-h1)/2.)), :]
    elif h1 > h2:
        im1 = im1[int(np.floor((h1-h2)/2.)) : -int(np.ceil((h1-h2)/2.)), :]
    if w1 < w2:
        im2 = im2[:, int(np.floor((w2-w1)/2.)) : -int(np.ceil((w2-w1)/2.))]
    elif w1 > w2:
        im1 = im1[:, int(np.floor((w1-w2)/2.)) : -int(np.ceil((w1-w2)/2.))]
    assert im1.shape == im2.shape
    return im1, im2

def align_images(im1: np.ndarray, im2: np.ndarray) -> tuple:
    pts = get_points(im1, im2)
    im1, im2 = align_image_centers(im1, im2, pts)
    im1, im2 = rescale_images(im1, im2, pts)
    im1, angle = rotate_im1(im1, pts)
    im1, im2 = match_img_size(im1, im2)
    return im1, im2

def load_local_image(filename: str) -> np.ndarray:
    img = skio.imread(filename)
    if img.dtype == np.uint8:
        img = img.astype(np.float64) / 255.0
    return img

def get_gaussian_kernel2d(ksize: int, sigma: float) -> np.ndarray:
    # 2d isotropic gaussian kernel
    ax = np.linspace(-(ksize // 2), ksize // 2, ksize)
    kernel1d = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel1d = kernel1d / np.sum(kernel1d)
    return np.outer(kernel1d, kernel1d)

def compute_fft_spectrum(img: np.ndarray) -> np.ndarray:
    # 2d fourier transform log magnitude spectrum
    if img.ndim == 3:
        img_gray = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
    else:
        img_gray = img

    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    return np.log(np.abs(fshift) + 1e-8)

def make_hybrid_image(img_low: np.ndarray, img_high: np.ndarray, sigma_low: float, sigma_high: float) -> tuple:
    # filter low and high freqs and combine them 
    ksize_low = int(6 * sigma_low + 1) | 1
    ksize_high = int(6 * sigma_high + 1) | 1

    g_low = get_gaussian_kernel2d(ksize_low, sigma_low)
    g_high = get_gaussian_kernel2d(ksize_high, sigma_high)

    if img_low.ndim == 3:
        low_pass = np.zeros_like(img_low)
        high_pass = np.zeros_like(img_high)
        for c in range(img_low.shape[2]):
            low_pass[:, :, c] = scipy.signal.convolve2d(img_low[:, :, c], g_low, mode='same', boundary='symm')
            hp_blur = scipy.signal.convolve2d(img_high[:, :, c], g_high, mode='same', boundary='symm')
            high_pass[:, :, c] = img_high[:, :, c] - hp_blur
    else:
        low_pass = scipy.signal.convolve2d(img_low, g_low, mode='same', boundary='symm')
        hp_blur = scipy.signal.convolve2d(img_high, g_high, mode='same', boundary='symm')
        high_pass = img_high - hp_blur

    hybrid = low_pass + high_pass
    return np.clip(hybrid, 0, 1), np.clip(low_pass, 0, 1), high_pass

def question_2_2(img1, img2, prefix):
    im1 = load_local_image(img1)   # low freq target
    im2 = load_local_image(img2)  # high freq target

    # align image via point selection
    print("\nClick 2 points on Image 1 (e.g. eyes), then 2 points on Image 2.")
    plt.ion()
    im1_aligned, im2_aligned = align_images(im1, im2)
    plt.ioff()

    # create hybrid image
    sigma_low = 6.0   
    sigma_high = 3.0
    hybrid, low_pass, high_pass = make_hybrid_image(im1_aligned, im2_aligned, sigma_low, sigma_high)

    # compute 2d fourier magnitude spectra
    fft_im1 = compute_fft_spectrum(im1_aligned)
    fft_im2 = compute_fft_spectrum(im2_aligned)
    fft_low = compute_fft_spectrum(low_pass)
    fft_high = compute_fft_spectrum(high_pass)
    fft_hybrid = compute_fft_spectrum(hybrid)

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))

    axes[0, 0].imshow(im1_aligned); axes[0, 0].set_title("Aligned Img 1")
    axes[0, 1].imshow(im2_aligned); axes[0, 1].set_title("Aligned Img 2")
    axes[0, 2].imshow(low_pass); axes[0, 2].set_title(fr"Low-Pass ($\sigma={sigma_low}$)")
    axes[0, 3].imshow(np.clip((high_pass + 1.0) / 2.0, 0, 1)); axes[0, 3].set_title(fr"High-Pass ($\sigma={sigma_high}$)")
    axes[0, 4].imshow(hybrid); axes[0, 4].set_title("Hybrid Image")

    # frequency row
    axes[1, 0].imshow(fft_im1, cmap='gray'); axes[1, 0].set_title("FFT: Input 1")
    axes[1, 1].imshow(fft_im2, cmap='gray'); axes[1, 1].set_title("FFT: Input 2")
    axes[1, 2].imshow(fft_low, cmap='gray'); axes[1, 2].set_title("FFT: Low-Pass")
    axes[1, 3].imshow(fft_high, cmap='gray'); axes[1, 3].set_title("FFT: High-Pass")
    axes[1, 4].imshow(fft_hybrid, cmap='gray'); axes[1, 4].set_title("FFT: Hybrid")

    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig("part2_2_" + prefix + "_hybrid_fourier.png", dpi=300, bbox_inches='tight')
    plt.show()

    skio.imsave("part2_2_" + prefix + "_hybrid_result.png", (hybrid * 255).astype(np.uint8))

# question 2.3:
def build_gaussian_stack(img: np.ndarray, num_levels: int, sigma_init: float = 2.0) -> list:
    g_stack = [img.copy()]
    
    for l in range(1, num_levels):
        sigma = sigma_init * (2.0 ** l)
        ksize = int(6 * sigma + 1) | 1  # Ensure kernel size is odd
        blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
        g_stack.append(blurred)

    return g_stack

def build_laplacian_stack(g_stack: list) -> list:
    l_stack = []
    num_levels = len(g_stack)

    for l in range(num_levels - 1):
        lap = g_stack[l] - g_stack[l + 1]
        l_stack.append(lap)

    l_stack.append(g_stack[-1].copy())  # Base low-frequency residual level
    return l_stack

def reconstruct_from_laplacian(l_stack: list) -> np.ndarray:
    reconstructed = np.sum(l_stack, axis=0)
    return np.clip(reconstructed, 0.0, 1.0)

def question_2_3():
    apple = load_local_image('apple.jpeg')
    orange = load_local_image('orange.jpeg')

    H = min(apple.shape[0], orange.shape[0])
    W = min(apple.shape[1], orange.shape[1])
    apple = cv2.resize(apple, (W, H))
    orange = cv2.resize(orange, (W, H))

    # create vertical mask
    vertical_mask = np.zeros((H, W), dtype=np.float64)
    vertical_mask[:, :W // 2] = 1.0

    num_levels = 5

    # building the stacks 
    g_apple = build_gaussian_stack(apple, num_levels)
    g_orange = build_gaussian_stack(orange, num_levels)
    g_mask = build_gaussian_stack(vertical_mask, num_levels)

    l_apple = build_laplacian_stack(g_apple)
    l_orange = build_laplacian_stack(g_orange)

    # computing the blended levels across the laplacian stack 
    l_blend = []
    for l in range(num_levels):
        m = np.expand_dims(g_mask[l], axis=-1) if g_mask[l].ndim == 2 else g_mask[l]
        blend_l = m * l_apple[l] + (1.0 - m) * l_orange[l]
        l_blend.append(blend_l)

    # reconstructing oraple
    oraple = reconstruct_from_laplacian(l_blend)

    # plotting breakdown 
    fig, axes = plt.subplots(num_levels, 6, figsize=(18, 3 * num_levels))

    for l in range(num_levels):
        disp_g_apple = np.clip(g_apple[l], 0, 1)
        disp_g_orange = np.clip(g_orange[l], 0, 1)

        if l < num_levels - 1:
            disp_l_apple = np.clip(l_apple[l] + 0.5, 0, 1)
            disp_l_orange = np.clip(l_orange[l] + 0.5, 0, 1)
            disp_l_blend = np.clip(l_blend[l] + 0.5, 0, 1)
        else:
            disp_l_apple = np.clip(l_apple[l], 0, 1)
            disp_l_orange = np.clip(l_orange[l], 0, 1)
            disp_l_blend = np.clip(l_blend[l], 0, 1)

        axes[l, 0].imshow(disp_g_apple)
        axes[l, 0].set_ylabel(f"Level {l}", fontsize=12)
        axes[l, 1].imshow(disp_g_orange)
        axes[l, 2].imshow(disp_l_apple)
        axes[l, 3].imshow(disp_l_orange)
        axes[l, 4].imshow(g_mask[l], cmap='gray')
        axes[l, 5].imshow(disp_l_blend)

    # column titles
    titles = [
        "Gaussian Apple", "Gaussian Orange",
        "Laplacian Apple", "Laplacian Orange",
        "Gaussian Mask", "Blended Laplacian"
    ]
    for col_idx, title in enumerate(titles):
        axes[0, col_idx].set_title(title, fontsize=11, fontweight='bold')

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("part2_3_stack_visualization.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 5. Save and display final reconstructed Oraple
    plt.figure(figsize=(6, 6))
    plt.imshow(oraple)
    plt.title("Final Reconstructed Oraple", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("part2_3_oraple_reconstructed.png", dpi=300, bbox_inches='tight')
    plt.show() 

# question 2.4: 
def load_and_preprocess(pathA: str, pathB: str) -> tuple[np.ndarray, np.ndarray]:
    # load two images, convert to float, resize image b to match image a
    imgA = skio.imread(pathA)
    imgB = skio.imread(pathB)

    if imgA.dtype == np.uint8: imgA = imgA.astype(np.float64) / 255.0
    if imgB.dtype == np.uint8: imgB = imgB.astype(np.float64) / 255.0

    # Ensure 3 channels for consistency
    if imgA.ndim == 2: imgA = np.stack([imgA] * 3, axis=-1)
    if imgB.ndim == 2: imgB = np.stack([imgB] * 3, axis=-1)
    
    # Drop alpha channel if present
    imgA = imgA[:, :, :3]
    imgB = imgB[:, :, :3]

    # Resize imgB to match imgA's dimensions
    H, W = imgA.shape[:2]
    imgB = cv2.resize(imgB, (W, H))

    return imgA, imgB

def multiresolution_blend(imgA: np.ndarray, imgB: np.ndarray, mask: np.ndarray, num_levels: int = 5) -> np.ndarray:
    g_stack_A = build_gaussian_stack(imgA, num_levels)
    g_stack_B = build_gaussian_stack(imgB, num_levels)
    g_stack_M = build_gaussian_stack(mask, num_levels)

    l_stack_A = build_laplacian_stack(g_stack_A)
    l_stack_B = build_laplacian_stack(g_stack_B)

    l_stack_blend = []
    for l in range(num_levels):
        m = np.expand_dims(g_stack_M[l], axis=-1) if (imgA.ndim == 3 and mask.ndim == 2) else g_stack_M[l]
        blend_l = m * l_stack_A[l] + (1.0 - m) * l_stack_B[l]
        l_stack_blend.append(blend_l)

    return np.clip(np.sum(l_stack_blend, axis=0), 0, 1)
# generating mask 
def create_straight_mask(shape: tuple, orientation: str = 'vertical') -> np.ndarray:
    # generate vertical or horizontal mask 
    H, W = shape[:2]
    mask = np.zeros((H, W), dtype=np.float64)
    if orientation == 'vertical':
        mask[:, :W // 2] = 1.0
    elif orientation == 'horizontal':
        mask[:H // 2, :] = 1.0
    return mask

def interactive_polygon_mask(img: np.ndarray, max_display_width: int = 1280, max_display_height: int = 720) -> np.ndarray:
    """
    Opens a resizable OpenCV window tailored to fit your screen.
    
    Controls:
    - Left Click   : Add vertex
    - 'r'          : Reset/clear points
    - 'c' or ENTER : Confirm mask selection
    """
    H, W = img.shape[:2]
    pts = []

    # Prepare display image (RGB -> BGR uint8)
    display_img = (img * 255).astype(np.uint8)
    if display_img.ndim == 3:
        display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
    else:
        display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)

    window_name = "Select Mask Area | 'c'=Confirm | 'r'=Reset"

    def draw_callback(event, x, y, flags, param):
        nonlocal pts
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))

    # Enable manual window resizing & scale to fit screen dimensions
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    scale = min(max_display_width / W, max_display_height / H, 1.0)
    disp_w, disp_h = int(W * scale), int(H * scale)
    cv2.resizeWindow(window_name, disp_w, disp_h)

    cv2.setMouseCallback(window_name, draw_callback)

    while True:
        canvas = display_img.copy()

        # Render points and lines on original full-resolution canvas
        if len(pts) > 0:
            for pt in pts:
                cv2.circle(canvas, pt, max(3, int(5 / scale)), (0, 0, 255), -1)

            if len(pts) > 1:
                cv2.polylines(canvas, [np.array(pts, np.int32)], isClosed=False, color=(0, 255, 0), thickness=max(1, int(2 / scale)))

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('r'):
            pts = []
            print("Points cleared. Start clicking again.")
        elif key == ord('c') or key == 13:
            if len(pts) < 3:
                print("Please select at least 3 points to create a mask.")
            else:
                break

    cv2.destroyWindow(window_name)

    # Generate full-resolution float mask [0.0, 1.0]
    mask = np.zeros((H, W), dtype=np.uint8)
    pts_array = np.array(pts, dtype=np.int32)
    cv2.fillPoly(mask, [pts_array], 255)
    
    return mask.astype(np.float64) / 255.0

def blend_with_interactive_mask(pathA: str, pathB: str, output_prefix: str, num_levels: int = 5):
    imgA, imgB = load_and_preprocess(pathA, pathB)

    # interactive roi selection on image a
    print("1. Click points around the feature on Image A to form your mask")
    print("2. Press 'c' or ENTER when finished")

    custom_mask = interactive_polygon_mask(imgA)

    blended = multiresolution_blend(imgA, imgB, custom_mask, num_levels=num_levels)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(imgA); ax[0].set_title("Image A")
    ax[1].imshow(imgB); ax[1].set_title("Image B")
    ax[2].imshow(custom_mask, cmap='gray'); ax[2].set_title("Irregular Mask")
    ax[3].imshow(blended); ax[3].set_title("Blended Result")
    for a in ax: a.axis('off')
    plt.tight_layout()
    plt.savefig(f"part2_4_{output_prefix}_summary.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Save output image
    skio.imsave(f"part2_4_{output_prefix}_result.png", (blended * 255).astype(np.uint8))

def save_figure_10_breakdown(imgA: np.ndarray, imgB: np.ndarray, mask: np.ndarray, num_levels: int, save_path: str):
    g_stack_A = build_gaussian_stack(imgA, num_levels)
    g_stack_B = build_gaussian_stack(imgB, num_levels)
    g_stack_M = build_gaussian_stack(mask, num_levels)

    l_stack_A = build_laplacian_stack(g_stack_A)
    l_stack_B = build_laplacian_stack(g_stack_B)

    fig, axes = plt.subplots(3, num_levels, figsize=(3 * num_levels, 8))

    for l in range(num_levels):
        m = np.expand_dims(g_stack_M[l], axis=-1) if (imgA.ndim == 3 and mask.ndim == 2) else g_stack_M[l]

        compA = np.clip(m * l_stack_A[l] + 0.5, 0, 1)
        compB = np.clip((1.0 - m) * l_stack_B[l] + 0.5, 0, 1)
        comp_blend = np.clip((m * l_stack_A[l] + (1.0 - m) * l_stack_B[l]) + 0.5, 0, 1)

        axes[0, l].imshow(compA); axes[0, l].set_title(f"A Level {l}"); axes[0, l].axis('off')
        axes[1, l].imshow(compB); axes[1, l].set_title(f"B Level {l}"); axes[1, l].axis('off')
        axes[2, l].imshow(comp_blend); axes[2, l].set_title(f"Blend Level {l}"); axes[2, l].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def blend_custom_pair(
    pathA: str, 
    pathB: str, 
    mask: np.ndarray, 
    output_prefix: str, 
    num_levels: int = 5, 
    generate_fig10: bool = False
):
    imgA, imgB = load_and_preprocess(pathA, pathB)
    
    blended = multiresolution_blend(imgA, imgB, mask, num_levels=num_levels)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(imgA); ax[0].set_title("Image A")
    ax[1].imshow(imgB); ax[1].set_title("Image B")
    ax[2].imshow(mask, cmap='gray'); ax[2].set_title("Mask")
    ax[3].imshow(blended); ax[3].set_title("Blended Result")
    for a in ax: a.axis('off')
    plt.tight_layout()
    plt.savefig(f"part2_4_{output_prefix}_summary.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Save full resolution output
    skio.imsave(f"part2_4_{output_prefix}_result.png", (blended * 255).astype(np.uint8))

    # Generate Figure 10 if requested
    if generate_fig10:
        save_figure_10_breakdown(imgA, imgB, mask, num_levels, f"{output_prefix}_fig10.png")

def question_2_4():
    # straight seam blend (Oraple)
    imgA = "apple.jpeg"
    imgB = "orange.jpeg"

    temp_A, _ = load_and_preprocess(imgA, imgB)
    mask_straight = create_straight_mask(temp_A.shape, orientation='vertical')
    
    blend_custom_pair(
        pathA=imgA, 
        pathB=imgB, 
        mask=mask_straight, 
        output_prefix="oraple",
        num_levels=5
    )

    # irregular mask blend (candy and flower) 
    blend_with_interactive_mask(
        pathA="candy.jpg", 
        pathB="poppy.jpg", 
        output_prefix="candy_flower", 
        num_levels=5
    )

    # irregular mask blend (two lakes reflection)
    blend_with_interactive_mask(
        pathA="lake_lassen.jpg", 
        pathB="lake_matheson.jpg", 
        output_prefix="lake_reflection", 
        num_levels=5
    )    
        
# run all the questions 
def main(): 
    question_1_1()
    question_1_2()
    question_1_3()
    question_2_1()
    question_2_2("joe_bruin.jpg","oski.jpeg","1")
    question_2_2("kitten.jpg","rumtumtugger.jpg","2")
    question_2_3()
    question_2_4()

main() 
