# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import configargparse
from tqdm import tqdm
import torch
from torchvision import transforms
from PIL import Image, ImageFilter
import random
import cv2
from sklearn import metrics

def mask_with_pose(image, pose, mean_dis, min_f, max_f):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    T = pose[:3, 3]
    Rz = pose[:3, 2]
    z_projection_length = np.dot(T, Rz)
    dis = abs(z_projection_length)

    short_len = min(rows, cols)
    min_f = min_f / mean_dis * dis
    max_f = max_f / mean_dis * dis
    mask = np.ones((rows, cols), np.uint8)

    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow)**2 + (j - ccol)**2)
            if min_f <= distance <= max_f:
                mask[i, j] = 0
    return mask

def calculate_mean_with_mask(data, mask):
    return np.mean(data * (1 - mask))

def read_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def read_setting(filepath):
    with open(filepath, 'r') as f:
        settings = json.load(f)
    return settings['mean_dis'], settings['min_f'], settings['max_f']

def image_spectrum(self, image, save_path=None):
    _, _, c = image.shape

    fft_all_channels = np.zeros_like(image, dtype=np.complex)

    for channel in range(c):

        single_channel = image[:, :, channel]

        f_transform = np.fft.fft2(single_channel)
        f_transform_shifted = np.fft.fftshift(f_transform)

        fft_all_channels[:, :, channel] = np.abs(f_transform_shifted)

        if save_path:

            magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted) + 1)

            magnitude_spectrum_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
            magnitude_spectrum_uint8 = magnitude_spectrum_normalized.astype(np.uint8)

            save_path_fft = f'{os.path.splitext(save_path)[0]}_channel_{channel}.png'

            cv2.imwrite(save_path_fft, magnitude_spectrum_uint8)

    return fft_all_channels

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_and_save_spectrum(image, save_path=None, image_file=None):
    _, _, c = image.shape
    combined_spectrum = []

    for channel in range(c):

        single_channel = image[:, :, channel]

        f_transform = np.fft.fft2(single_channel)
        f_transform_shifted = np.fft.fftshift(f_transform)

        magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted) + 1)
        phase_spectrum = np.angle(f_transform_shifted)

        combined_spectrum.append(np.abs(f_transform_shifted))

    combined_spectrum_np = np.stack(combined_spectrum, axis=-1)
    
    return combined_spectrum_np

def center_crop_with_padding(image_pil, scale, original_size):
    """Center crop the image and add white padding to maintain original size"""
    # Calculate target crop size
    crop_width = int(image_pil.size[0] * scale)
    crop_height = int(image_pil.size[1] * scale)
    
    # Create center crop transform
    center_crop = transforms.CenterCrop((crop_height, crop_width))
    cropped_image = center_crop(image_pil)
    
    # Create new white background image
    padded_image = Image.new('RGB', original_size, (255, 255, 255))
    
    # Calculate paste position (center)
    paste_x = (original_size[0] - crop_width) // 2
    paste_y = (original_size[1] - crop_height) // 2
    
    # Paste cropped image onto white background
    padded_image.paste(cropped_image, (paste_x, paste_y))
    
    return padded_image

def rotate_with_padding(image_pil, angle):
    """Rotate image and maintain original size by adding white padding"""
    # Get original size
    original_size = image_pil.size
    
    # Rotate image
    rotated_image = image_pil.rotate(angle, expand=True, fillcolor=(255, 255, 255))
    
    # Create new white background image with original size
    final_image = Image.new('RGB', original_size, (255, 255, 255))
    
    # Calculate position to paste rotated image
    paste_x = (original_size[0] - rotated_image.size[0]) // 2
    paste_y = (original_size[1] - rotated_image.size[1]) // 2
    
    # Paste rotated image onto white background
    final_image.paste(rotated_image, (paste_x, paste_y))
    
    return final_image

def apply_distortions(image, args, save_dir, index, seed=None, save_images=False):
    """Apply various distortions to an image and save results"""
    from PIL import Image, ImageFilter
    import torchvision.transforms as transforms
    import numpy as np
    import os
    
    # Only create directories if save_images is True
    if save_images:
        distorted_save_dir = os.path.join(save_dir, 'distorted_images')
        fft_save_dir = os.path.join(save_dir, 'fft_images')
        os.makedirs(distorted_save_dir, exist_ok=True)
        os.makedirs(fft_save_dir, exist_ok=True)
    
    # Convert OpenCV image to PIL image
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    original_size = image_pil.size
    
    # Dictionary to track applied distortions
    applied_distortions = {}
    
    if args.r_degree is not None:
        image_pil = rotate_with_padding(image_pil, args.r_degree)
        applied_distortions['rotation'] = args.r_degree
        
    if args.jpeg_ratio_1 is not None:
        temp_path = f"tmp_{args.jpeg_ratio_1}.jpg"
        image_pil.save(temp_path, quality=args.jpeg_ratio_1)
        image_pil = Image.open(temp_path)
        os.remove(temp_path)
        applied_distortions['jpeg1'] = args.jpeg_ratio_1
        
    if args.jpeg_ratio_2 is not None:
        temp_path = f"tmp_{args.jpeg_ratio_2}.jpg"
        image_pil.save(temp_path, quality=args.jpeg_ratio_2)
        image_pil = Image.open(temp_path)
        os.remove(temp_path)
        applied_distortions['jpeg2'] = args.jpeg_ratio_2
        
    if args.jpeg_ratio_3 is not None:
        temp_path = f"tmp_{args.jpeg_ratio_3}.jpg"
        image_pil.save(temp_path, quality=args.jpeg_ratio_3)
        image_pil = Image.open(temp_path)
        os.remove(temp_path)
        applied_distortions['jpeg3'] = args.jpeg_ratio_3
        
    if args.crop_scale is not None:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        image_pil = center_crop_with_padding(image_pil, args.crop_scale, original_size)
        applied_distortions['crop'] = args.crop_scale
        
    if args.gaussian_blur_r is not None:
        image_pil = image_pil.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))
        applied_distortions['blur'] = args.gaussian_blur_r
        
    if args.gaussian_std is not None:
        img_array = np.array(image_pil)
        g_noise = np.random.normal(0, args.gaussian_std, img_array.shape) * 255
        g_noise = g_noise.astype(np.uint8)
        image_pil = Image.fromarray(np.clip(img_array + g_noise, 0, 255))
        applied_distortions['noise'] = args.gaussian_std
        
    if args.brightness_factor is not None:
        image_pil = transforms.ColorJitter(brightness=args.brightness_factor)(image_pil)
        applied_distortions['brightness'] = args.brightness_factor
    
    # Convert back to OpenCV format
    distorted_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    if save_images:
        # Save distorted image
        distortion_name = '_'.join([f"{k}_{v}" for k, v in applied_distortions.items()])
        distorted_path = os.path.join(distorted_save_dir, f'distorted_{index}_{distortion_name}.png')
        cv2.imwrite(distorted_path, distorted_image)
    
        # Calculate and save FFT
        distorted_rgb = cv2.cvtColor(distorted_image, cv2.COLOR_BGR2RGB)
        for channel in range(3):
            single_channel = distorted_rgb[:, :, channel]
            f_transform = np.fft.fft2(single_channel)
            f_transform_shifted = np.fft.fftshift(f_transform)
            magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted) + 1)
            
            magnitude_spectrum_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
            magnitude_spectrum_uint8 = magnitude_spectrum_normalized.astype(np.uint8)
            
            fft_path = os.path.join(fft_save_dir, f'fft_{index}_{distortion_name}_channel_{channel}.png')
            cv2.imwrite(fft_path, magnitude_spectrum_uint8)
    
    return distorted_image

def create_single_distortion_args(dist_type, value):
    """Create a DistortionArgs object with only one distortion enabled"""
    class DistortionArgs:
        pass
    single_args = DistortionArgs()
    
    # Initialize all possible distortion attributes to None
    single_args.r_degree = None
    single_args.jpeg_ratio_1 = None
    single_args.jpeg_ratio_2 = None
    single_args.jpeg_ratio_3 = None
    single_args.crop_scale = None
    single_args.gaussian_blur_r = None
    single_args.gaussian_std = None
    single_args.brightness_factor = None
    
    # Set the specified distortion
    setattr(single_args, dist_type, value)
    return single_args

def read_images(original_results, watermarked_results, watermarked_dataset, n_test, distortion_args=None, save_images=False, detector='band_zero'):
    from sklearn import metrics
    mean_dis, min_f, max_f = read_setting(os.path.join(watermarked_dataset, "setting.json"))
    transforms = read_json(os.path.join(watermarked_dataset, "transforms_test_watermark.json"))
    print(f"Detector: {detector}")
    
    # Create save directory for distorted images if save_images is True
    save_dir = os.path.join(watermarked_results, "distortion_analysis")
    if save_images:
        os.makedirs(save_dir, exist_ok=True)
    
    # Dictionary to store results for clean and distorted images
    all_results = {
        'clean': {
            'original_means': [[], [], [], []],
            'watermarked_means': [[], [], [], []],
            'metrics': None
        }
    }
    
    # Create separate DistortionArgs for each distortion type
    distortion_configs = []
    if distortion_args:
        for dist_type in vars(distortion_args):
            value = getattr(distortion_args, dist_type)
            if value is not None:
                single_dist_args = create_single_distortion_args(dist_type, value)
                distortion_configs.append((dist_type, single_dist_args))
                all_results[dist_type] = {
                    'original_means': [[], [], [], []],
                    'watermarked_means': [[], [], [], []],
                    'metrics': None
                }

    # Process images
    for i in tqdm(range(n_test), desc="Processing images"):
        file_suffix = f"{i:06d}_rgb"
        pose = np.array(transforms['frames'][i]['transform_matrix'])
        
        # Process original images
        original_file = os.path.join(f"{original_results}/results/test", f"{file_suffix}.png")
        if not os.path.exists(original_file):
            print(f"File not found: {original_file}")
            continue
            
        original_image = cv2.imread(original_file)
        
        # Save original image FFT if save_images is True
        if save_images:
            original_fft_dir = os.path.join(save_dir, 'original_fft')
            os.makedirs(original_fft_dir, exist_ok=True)
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            for channel in range(3):
                single_channel = original_rgb[:, :, channel]
                f_transform = np.fft.fft2(single_channel)
                f_transform_shifted = np.fft.fftshift(f_transform)
                magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted) + 1)
                magnitude_spectrum_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
                magnitude_spectrum_uint8 = magnitude_spectrum_normalized.astype(np.uint8)
                fft_path = os.path.join(original_fft_dir, f'original_fft_{i}_channel_{channel}.png')
                cv2.imwrite(fft_path, magnitude_spectrum_uint8)
        
        # Process watermarked images
        watermarked_file = os.path.join(f"{watermarked_results}/results/test", f"{file_suffix}.png")
        if not os.path.exists(watermarked_file):
            print(f"File not found: {watermarked_file}")
            continue
            
        watermarked_image = cv2.imread(watermarked_file)
        
        # Save watermarked image FFT if save_images is True
        if save_images:
            watermarked_fft_dir = os.path.join(save_dir, 'watermarked_fft')
            os.makedirs(watermarked_fft_dir, exist_ok=True)
            watermarked_rgb = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2RGB)
            for channel in range(3):
                single_channel = watermarked_rgb[:, :, channel]
                f_transform = np.fft.fft2(single_channel)
                f_transform_shifted = np.fft.fftshift(f_transform)
                magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted) + 1)
                magnitude_spectrum_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
                magnitude_spectrum_uint8 = magnitude_spectrum_normalized.astype(np.uint8)
                fft_path = os.path.join(watermarked_fft_dir, f'watermarked_fft_{i}_channel_{channel}.png')
                cv2.imwrite(fft_path, magnitude_spectrum_uint8)
        
        # Process clean images
        process_image_pair(original_image, watermarked_image, pose, mean_dis, min_f, max_f, 
                         detector,
                         all_results['clean']['original_means'], 
                         all_results['clean']['watermarked_means'])
        
        # Process each distortion separately
        if distortion_configs:
            for dist_type, single_dist_args in distortion_configs:
                # Create directory for this specific distortion if save_images is True
                dist_dir = os.path.join(save_dir, dist_type) if save_images else None
                if save_images:
                    os.makedirs(dist_dir, exist_ok=True)
                
                # Apply distortion to original image
                orig_distorted = apply_distortions(
                    original_image.copy(), 
                    single_dist_args, 
                    dist_dir,
                    f"orig_{i}",
                    seed=i,
                    save_images=save_images
                )
                
                # Apply same distortion to watermarked image
                water_distorted = apply_distortions(
                    watermarked_image.copy(), 
                    single_dist_args, 
                    dist_dir,
                    f"water_{i}",
                    seed=i,
                    save_images=save_images
                )
                
                # Process distorted image pair
                process_image_pair(
                    orig_distorted,
                    water_distorted,
                    pose,
                    mean_dis,
                    min_f,
                    max_f,
                    detector,
                    all_results[dist_type]['original_means'],
                    all_results[dist_type]['watermarked_means']
                )

    # Calculate metrics for all conditions
    channel_names = ['R', 'G', 'B', 'Mean']
    metrics_summary = {}
    
    for condition in all_results:
        metrics_results = calculate_metrics(
            all_results[condition]['original_means'],
            all_results[condition]['watermarked_means']
        )
        all_results[condition]['metrics'] = metrics_results
        
        # Store metrics for this condition
        metrics_summary[condition] = {}
        
        # Print results
        print(f"\nResults for {condition}:")
        for i, channel in enumerate(channel_names):
            print(f"\n{channel} Channel:")
            print(f"AUC: {metrics_results['auc'][i]:.4f}")
            print(f"TPR@1%FPR: {metrics_results['tpr_at_1_fpr'][i]:.4f}")
            print(f"Accuracy: {metrics_results['accuracies'][i]:.4f}")
            
            metrics_summary[condition][channel] = {
                'auc': metrics_results['auc'][i],
                'tpr_at_1_fpr': metrics_results['tpr_at_1_fpr'][i],
                'accuracy': metrics_results['accuracies'][i]
            }

    # Save all results
    save_path = os.path.join(watermarked_results, "metrics_results_with_distortions.json")
    with open(save_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)

    return all_results

def create_ring_template(image_shape, pose, mean_dis, min_f, max_f):
    rows, cols = image_shape
    template = np.zeros((rows, cols), dtype=np.float32)
    crow, ccol = rows // 2, cols // 2
    

    T = pose[:3, 3]
    Rz = pose[:3, 2]
    z_projection_length = np.dot(T, Rz)
    dis = abs(z_projection_length)
    

    min_f_scaled = min_f / mean_dis * dis
    max_f_scaled = max_f / mean_dis * dis
    

    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow)**2 + (j - ccol)**2)
            if min_f_scaled <= distance <= max_f_scaled:

                template[i, j] = np.exp(-(distance - (min_f_scaled + max_f_scaled)/2)**2 
                                      / (2 * ((max_f_scaled - min_f_scaled)/4)**2))
    

    template = template / np.sqrt(np.sum(template**2))
    
    return template

def matched_filter_detection(spectrum_data, template):

    log_spectrum = np.log1p(np.abs(spectrum_data))
    

    normalized_spectrum = (log_spectrum - np.mean(log_spectrum)) / (np.std(log_spectrum) + 1e-10)
    

    correlation = cv2.matchTemplate(
        normalized_spectrum.astype(np.float32), 
        template.astype(np.float32), 
        cv2.TM_CCORR_NORMED
    )
    

    peak_value = np.max(correlation)
    
    return peak_value

def band_zero_detection(spectrum_data, template):
    """
    Detect the notch watermark by measuring how close the target band is to zero.
    Larger return values indicate a stronger watermark signal.
    """
    log_spectrum = np.log1p(np.abs(spectrum_data))
    weights = np.maximum(template.astype(np.float32), 0)
    weight_sum = np.sum(weights)
    if weight_sum <= 1e-12:
        return 0.0
    band_mse_to_zero = np.sum(weights * (log_spectrum ** 2)) / weight_sum
    return -float(band_mse_to_zero)

def detection_score(spectrum_data, template, detector):
    if detector == 'band_zero':
        return band_zero_detection(spectrum_data, template)
    if detector == 'matched_filter':
        return -matched_filter_detection(spectrum_data, template)
    raise ValueError(f"Unsupported detector: {detector}")

def process_image_pair(original_image, watermarked_image, pose, mean_dis, min_f, max_f, detector, original_means, watermarked_means):

    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    watermarked_image = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2RGB)
    

    original_data = plot_and_save_spectrum(original_image, None, None)
    watermarked_data = plot_and_save_spectrum(watermarked_image, None, None)
    

    template = create_ring_template(original_data.shape[:2], pose, mean_dis, min_f, max_f)
    
    orig_channel_means = []
    water_channel_means = []
    

    for channel in range(3):

        original_data_channel = original_data[:, :, channel]
        orig_score = detection_score(original_data_channel, template, detector)
        original_means[channel].append(orig_score)
        orig_channel_means.append(orig_score)
        

        watermarked_data_channel = watermarked_data[:, :, channel]
        water_score = detection_score(watermarked_data_channel, template, detector)
        watermarked_means[channel].append(water_score)
        water_channel_means.append(water_score)
    

    original_means[3].append(np.mean(orig_channel_means))
    watermarked_means[3].append(np.mean(water_channel_means))

def calculate_metrics(original_means, watermarked_means):
    """Helper function to calculate metrics for a set of means"""
    metrics_results = {
        'auc': [],
        'tpr_at_1_fpr': [],
        'accuracies': []
    }
    
    for channel in range(4):
        preds = original_means[channel] + watermarked_means[channel]
        t_labels = [0] * len(original_means[channel]) + [1] * len(watermarked_means[channel])
        
        fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        tpr_at_1_fpr = tpr[np.where(fpr <= 0.01)[0][-1]]
        acc = np.max(1 - (fpr + (1 - tpr))/2)
        
        metrics_results['auc'].append(auc)
        metrics_results['tpr_at_1_fpr'].append(tpr_at_1_fpr)
        metrics_results['accuracies'].append(acc)
    
    return metrics_results

def main():
    parser = configargparse.ArgumentParser(description="Read .npy files from original and watermarked results")
    
    # Required arguments
    parser.add_argument('--original_results', type=str, required=True, 
                       help='Path to original results directory')
    parser.add_argument('--watermarked_results', type=str, required=True, 
                       help='Path to watermarked results directory')
    parser.add_argument('--watermarked_dataset', type=str, required=True, 
                       help='Path to watermarked dataset directory')
    parser.add_argument('--n_test', type=int, required=True, 
                       help='Number of test files to read')
    
    # Optional distortion parameters
    parser.add_argument('--r_degree', type=float, default=None, 
                       help='Rotation degree')
    parser.add_argument('--jpeg_ratio_1', type=int, default=None, 
                       help='JPEG compression quality 1')
    parser.add_argument('--jpeg_ratio_2', type=int, default=None, 
                       help='JPEG compression quality 2')
    parser.add_argument('--jpeg_ratio_3', type=int, default=None, 
                       help='JPEG compression quality 3')
    parser.add_argument('--crop_scale', type=float, default=None, 
                       help='Center crop scale (0-1)')
    parser.add_argument('--gaussian_blur_r', type=float, default=None, 
                       help='Gaussian blur radius')
    parser.add_argument('--gaussian_std', type=float, default=None, 
                       help='Gaussian noise standard deviation')
    parser.add_argument('--brightness_factor', type=float, default=None, 
                       help='Brightness adjustment factor')
    
    # Save images flag
    parser.add_argument('--save_images', action='store_true', default=False,
                       help='Save distorted images and FFT visualizations')
    parser.add_argument('--detector', type=str, default='band_zero',
                       choices=['band_zero', 'matched_filter'],
                       help='Detection score. band_zero is the main method; matched_filter keeps the previous MF score.')

    args = parser.parse_args()

    # Validate path arguments
    if not os.path.exists(args.original_results):
        raise ValueError(f"Original results path does not exist: {args.original_results}")
    if not os.path.exists(args.watermarked_results):
        raise ValueError(f"Watermarked results path does not exist: {args.watermarked_results}")
    if not os.path.exists(args.watermarked_dataset):
        raise ValueError(f"Watermarked dataset path does not exist: {args.watermarked_dataset}")

    # Create distortion parameters if any distortion argument is provided
    distortion_args = None
    if any(getattr(args, attr) is not None for attr in [
        'r_degree', 'jpeg_ratio_1', 'jpeg_ratio_2', 'jpeg_ratio_3',
        'crop_scale', 'gaussian_blur_r', 'gaussian_std', 'brightness_factor'
    ]):
        class DistortionArgs:
            pass
        distortion_args = DistortionArgs()
        distortion_args.r_degree = args.r_degree
        distortion_args.jpeg_ratio_1 = args.jpeg_ratio_1
        distortion_args.jpeg_ratio_2 = args.jpeg_ratio_2
        distortion_args.jpeg_ratio_3 = args.jpeg_ratio_3
        distortion_args.crop_scale = args.crop_scale
        distortion_args.gaussian_blur_r = args.gaussian_blur_r
        distortion_args.gaussian_std = args.gaussian_std
        distortion_args.brightness_factor = args.brightness_factor

        # Validate distortion parameters
        if args.r_degree is not None and (args.r_degree < -180 or args.r_degree > 180):
            raise ValueError("Rotation degree must be between -180 and 180")
        if any(ratio is not None and (ratio < 0 or ratio > 100) 
               for ratio in [args.jpeg_ratio_1, args.jpeg_ratio_2, args.jpeg_ratio_3]):
            raise ValueError("JPEG quality must be between 0 and 100")
        if args.crop_scale is not None and (args.crop_scale <= 0 or args.crop_scale > 1):
            raise ValueError("Crop scale must be between 0 and 1")
        if args.gaussian_blur_r is not None and args.gaussian_blur_r < 0:
            raise ValueError("Gaussian blur radius must be non-negative")
        if args.gaussian_std is not None and args.gaussian_std < 0:
            raise ValueError("Gaussian noise standard deviation must be non-negative")
        if args.brightness_factor is not None and args.brightness_factor < 0:
            raise ValueError("Brightness factor must be non-negative")

    try:
        print("Starting image processing...")
        if distortion_args:
            print("Applied distortions:")
            for attr in vars(distortion_args):
                value = getattr(distortion_args, attr)
                if value is not None:
                    print(f"  {attr}: {value}")
        
        # Call read_images with all parameters
        results = read_images(
            args.original_results,
            args.watermarked_results,
            args.watermarked_dataset,
            args.n_test,
            distortion_args,
            save_images=args.save_images,
            detector=args.detector
        )
        
        print("\nProcessing completed successfully!")
        if args.save_images:
            print(f"Results and images saved in: {args.watermarked_results}")
        else:
            print(f"Results saved in: {args.watermarked_results}")
            
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
    
