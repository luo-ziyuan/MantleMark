# -*- coding: utf-8 -*-
import configargparse
import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
from nerf.utils import *
from tqdm import tqdm

def config_parser():
    parser = configargparse.ArgumentParser(description='PyTorch CIFAR100 Training')
    parser.add_argument('--input_data_path', type=str, required=True, help='input data path')
    parser.add_argument('--output_data_path', type=str, required=True, help='output data path')
    parser.add_argument('--min_f', type=float, default=70, help='Minimum frequency')
    parser.add_argument('--max_f', type=float, default=100, help='Maximum frequency')
    parser.add_argument('--watermark_strength', type=float, default=1.0, help='watermark_strength')

    args = parser.parse_args()
    return args

def plot_and_save_spectrum(image, save_path=None, image_file=None):
    _, _, c = image.shape
    for channel in range(c):

        single_channel = image[:, :, channel]

        f_transform = np.fft.fft2(single_channel)
        f_transform_shifted = np.fft.fftshift(f_transform)

        magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted) + 1)
        phase_spectrum = np.angle(f_transform_shifted)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(magnitude_spectrum, cmap='viridis')
        plt.title('amplitude')
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.imshow(phase_spectrum, cmap='viridis')
        plt.title('phase')
        plt.colorbar()

        plt.suptitle('Fourier Transform')
        
        if save_path:
            save_path_fft = os.path.join(save_path, f'{os.path.splitext(image_file)[0]}_channel_{channel}.png')
            plt.savefig(save_path_fft)
        plt.close()

def get_plane_coordinates(pose_matrix, size=800):

    pose = np.array(pose_matrix).reshape(4, 4)
    camera_position = pose[:3, 3]
    camera_direction = pose[:3, 2]

    view_direction = camera_direction
    view_direction /= np.linalg.norm(view_direction)

    up = np.array([0, 1, 0])
    if np.allclose(view_direction, up) or np.allclose(view_direction, -up):
        up = np.array([1, 0, 0])
    right = np.cross(view_direction, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, view_direction)
    up /= np.linalg.norm(up)

    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)

    points = np.zeros((size, size, 3))
    for i in range(size):
        for j in range(size):
            points[i, j] = xx[i, j] * right + yy[i, j] * up + camera_position

    return points

def is_point_in_ellipsoid(point, center, axes):
    x, y, z = point - center
    a, b, c = axes
    return (x/a)**2 + (y/b)**2 + (z/c)**2 <= 1

def generate_ellipsoid_mask(points, inner_axes, outer_axes, center):
    size = points.shape[0]
    mask = np.ones((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            point = points[i, j]
            if is_point_in_ellipsoid(point, center, outer_axes) and not is_point_in_ellipsoid(point, center, inner_axes):
                mask[i, j] = 0
    return mask

def create_ring_watermark(rows, cols, pose, mean_dis, min_f, max_f):
    mask = np.zeros((rows, cols), np.float32)
    crow, ccol = rows // 2, cols // 2
    

    T = pose[:3, 3]
    Rz = pose[:3, 2]
    z_projection_length = np.dot(T, Rz)
    dis = abs(z_projection_length)
    
    min_f_scaled = min_f / mean_dis * dis
    max_f_scaled = max_f / mean_dis * dis
    

    ring_center = (min_f_scaled + max_f_scaled) / 2
    ring_width = (max_f_scaled - min_f_scaled) / 2
    

    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow)**2 + (j - ccol)**2)

            mask[i, j] = np.exp(-(distance - ring_center)**2 / (2 * (ring_width/2)**2))
    
    return mask

def freq_modify(image):

    rows, cols, c = image.shape
    crow, ccol = rows // 2, cols // 2
    
    short_len = min(rows, cols)
    min_f = 70
    max_f = 100
    interval = int(float(max_f - min_f) / 3)

    mask = np.ones((rows, cols), np.uint8)
    
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow)**2 + (j - ccol)**2)
#             if (distance >= min_f and distance <= (min_f + interval)) or (distance >= (max_f - interval) and distance <= max_f) :
            if distance >= min_f and distance <= max_f :
                mask[i, j] = 0
    # save_path = os.path.join("./test.png")
    # cv2.imwrite(save_path, mask*255)         

    # for i in range(rows):
    #     for j in range(cols):
    #         distance = np.sqrt((i - crow)**2 + (j - ccol)**2)
    #         if 70 <= distance <= 100:
    #             phase_spectrum[i, j] = 0
    modified_image = np.zeros_like(image)
    # mask = np.expand_dims(mask, axis=2)
    for channel in range(c):

        single_channel = image[:, :, channel]

        f_transform = np.fft.fft2(single_channel)
        f_transform_shifted = np.fft.fftshift(f_transform)

        magnitude_spectrum = np.abs(f_transform_shifted)
        phase_spectrum = np.angle(f_transform_shifted)
        magnitude_spectrum = magnitude_spectrum * mask
        # phase_spectrum = phase_spectrum * mask

        f_transform_modified = magnitude_spectrum*np.exp(1j*phase_spectrum)

        f_transform_shifted = np.fft.ifftshift(f_transform_modified)
        new_single_channel = np.fft.ifft2(f_transform_shifted)
        new_single_channel = np.abs(new_single_channel)
        new_single_channel = np.clip(new_single_channel, 0, 255)
        modified_image[:, :, channel] = new_single_channel

    return modified_image

def freq_modify_with_pose(image, pose, mean_dis, args):
    rows, cols, c = image.shape
    

    mask = create_ring_watermark(rows, cols, pose, mean_dis, args.min_f, args.max_f)
    

    watermark_strength = args.watermark_strength
    
    modified_image = np.zeros_like(image)
    
    for channel in range(c):

        single_channel = image[:, :, channel]
        

        f_transform = np.fft.fft2(single_channel)
        f_transform_shifted = np.fft.fftshift(f_transform)
        

        magnitude_spectrum = np.abs(f_transform_shifted)
        phase_spectrum = np.angle(f_transform_shifted)
        

        magnitude_spectrum = magnitude_spectrum * (1 - watermark_strength * mask)
        

        f_transform_modified = magnitude_spectrum * np.exp(1j * phase_spectrum)
        

        f_transform_shifted = np.fft.ifftshift(f_transform_modified)
        new_single_channel = np.fft.ifft2(f_transform_shifted)
        new_single_channel = np.abs(new_single_channel)
        

        modified_image[:, :, channel] = np.clip(new_single_channel, 0, 255)
    
    return modified_image

def watermark(image):

    rows, cols, c = image.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.zeros((rows, cols), np.float32)
    
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow)**2 + (j - ccol)**2)
            if distance >= 70 and distance <= 100:
                mask[i, j] = 1
    
    watermark_pattern = np.zeros_like(image)
#     mask = np.clip(mask*255, 0, 255)
    for channel in range(c):
        img_back = np.real(np.fft.ifft2(np.fft.ifftshift(mask)))
        watermark_pattern[:, :, channel] = np.fft.fftshift(np.uint8(255 * (img_back - np.min(img_back)) / (np.max(img_back) - np.min(img_back))))

    return watermark_pattern

def log(self, *args, **kwargs):
    if self.local_rank == 0:
        if not self.mute: 
            #print(*args)
            self.console.print(*args, **kwargs)
        if self.log_ptr: 
            print(*args, file=self.log_ptr)
            self.log_ptr.flush() # write immediately to file

def watermark_freq(image):

    rows, cols, c = image.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.ones((rows, cols), np.float32)
    
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow)**2 + (j - ccol)**2)
            if distance >= 70 and distance <= 100:
                mask[i, j] = 0
    
    watermark_pattern = np.zeros_like(image)
    mask = np.clip(mask*255, 0, 255)
    for channel in range(c):
        watermark_pattern[:, :, channel] = mask

    return watermark_pattern

def calculate_average_z_projection_length(data):
    z_projection_lengths = []
    
    for frame in data['frames']:
        pose = frame['transform_matrix']
        
        # Convert the transform matrix to a numpy array for easier manipulation
        pose_matrix = np.array(pose)
        
        # Extract the translation vector T
        T = pose_matrix[:3, 3]
        
        # Extract the third column of the rotation matrix (camera's z-axis direction)
        Rz = pose_matrix[:3, 2]
        
        # Calculate the projection length of T on Rz
        z_projection_length = np.dot(T, Rz)
        
        # Take the absolute value of the projection length
        z_projection_length = abs(z_projection_length)
        
        # Append the absolute projection length to the list
        z_projection_lengths.append(z_projection_length)
    
    # Calculate the mean absolute projection length
    mean_z_projection_length = np.mean(z_projection_lengths)
    
    return mean_z_projection_length

def w_generator(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metrics = [PSNRMeter(), LPIPSMeter(device=device), SSIMMeter()]
    input_data_path = args.input_data_path
    output_data_path = args.output_data_path
    
    if os.path.exists(output_data_path):
        output_data_path_base = output_data_path
        counter = 1
        while os.path.exists(output_data_path):
            output_data_path = f"{output_data_path_base}_{counter}"
            counter += 1

    os.makedirs(output_data_path)

    shutil.copytree(input_data_path, output_data_path, dirs_exist_ok=True)

    # for name in ["train"]:
    for name in ["train", "val", "test"]:
        # rgb_path_original = os.path.join(output_data_path, f"{name}_original")
        # pose_path_original = os.path.join(output_data_path, f"transforms_{name}_original.json")

        # rgb_path = os.path.join(output_data_path, f"{name}")
        pose_path = os.path.join(output_data_path, f"transforms_{name}.json")
        
        # os.rename(rgb_path, rgb_path_original)
        # os.rename(pose_path, pose_path_original)
    
        # os.makedirs(rgb_path)
        # os.makedirs(pose_path)
    
        with open(pose_path, 'r') as f:
            data = json.load(f)
        if name == "train":
            mean_dis = calculate_average_z_projection_length(data)
        for frame in tqdm(data['frames'], desc="Processing frames"):
            file_path = frame['file_path']
            
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(output_data_path, file_path.lstrip('./'))
            else:
                image_path = os.path.join(output_data_path, file_path.lstrip('./') + '.png')
            
            pose = frame['transform_matrix']
            pose_matrix = np.array(pose)
            if os.path.isfile(image_path):

                image = Image.open(image_path)
                if image.mode == 'RGBA':
                    r, g, b, alpha = image.split()
                    alpha = np.array(alpha).astype(np.float32) / 255.0
                    b = b*alpha + (1 - alpha)*255
                    g = g*alpha + (1 - alpha)*255
                    r = r*alpha + (1 - alpha)*255
                    b = b.astype(np.uint8)
                    g = g.astype(np.uint8)
                    r = r.astype(np.uint8)
                    image = cv2.merge([b, g, r])
                else:
                    image = cv2.imread(image_path)
            else:
                raise Exception(f"{image_path} not exist")
            image_w = freq_modify_with_pose(image, pose_matrix, mean_dis, args)
            image_w_tensor = torch.tensor(image_w).unsqueeze(0)  / 255.0 # [H, W, C] -> [1, C, H, W]
            image_tensor = torch.tensor(image).unsqueeze(0)  / 255.0 # [H, W, C] -> [1, C, H, W]
            for metric in metrics:
                metric.update(image_w_tensor, image_tensor)
            # save_path = file_path + '.png'
            cv2.imwrite(image_path, image_w)
            plot_and_save_spectrum(image_w, output_data_path, file_path.lstrip('./') + '.png')

    file_path = os.path.join(output_data_path, f"results.json")
    for metric in metrics:
        metric.write_json(file_path)
        metric.clear()

    data = {"mean_dis": mean_dis,
            "min_f": args.min_f,
            "max_f": args.max_f,
    }
    with open(os.path.join(output_data_path, f"setting.json"), 'w') as file:
        json.dump(data, file, indent=4)
#     shutil.copyfile(os.path.join(output_data_path, f"setting.json"), os.path.join(input_data_path, f"setting.json"))

if __name__ == '__main__':
    args = config_parser()

    # points = get_plane_coordinates(pose_matrix)

    # mask = generate_ellipsoid_mask(points, inner_axes, outer_axes, center)
    
    # save_path = os.path.join('./', 'test_ell.png')
    # cv2.imwrite(save_path, mask * 255)
    w_generator(args)
