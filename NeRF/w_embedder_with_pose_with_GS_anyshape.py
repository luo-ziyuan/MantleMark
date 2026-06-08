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
    parser.add_argument('--shape', type=str, default='ring', choices=['ring', 'cube', 'tetrahedron', 'octahedron'], help='Watermark shape: ring, cube, tetrahedron or octahedron')

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

def watermark_function_cube_shell_smooth(point_3d, min_f, max_f):
    x, y, z = point_3d
    

    cube_distance = max(abs(x), abs(y), abs(z))
    

    if min_f <= cube_distance <= max_f:

        cube_center = (min_f + max_f) / 2
        cube_width = (max_f - min_f) / 2
        return np.exp(-(cube_distance - cube_center)**2 / (2 * (cube_width/2)**2))
    else:
        return 0.0

def watermark_function_tetrahedron_shell_smooth(point_3d, min_f, max_f):
    x, y, z = point_3d
    

    normals = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ])
    

    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    

    point = np.array([x, y, z])
    distances = []
    
    for normal in normals:

        distance = abs(np.dot(point, normal))
        distances.append(distance)
    

    tetrahedron_distance = max(distances)
    

    if min_f <= tetrahedron_distance <= max_f:

        tetrahedron_center = (min_f + max_f) / 2
        tetrahedron_width = (max_f - min_f) / 2
        return np.exp(-(tetrahedron_distance - tetrahedron_center)**2 / (2 * (tetrahedron_width/2)**2))
    else:
        return 0.0

def watermark_function_octahedron_shell_smooth(point_3d, min_f, max_f):
    x, y, z = point_3d
    

    
    # for normal in normals:

    #     distance = abs(np.dot(point, normal))
    #     distances.append(distance)
    

    octahedron_distance = (abs(x) + abs(y) + abs(z))/1.46459
    

    if min_f <= octahedron_distance <= max_f:

        octahedron_center = (min_f + max_f) / 2
        octahedron_width = (max_f - min_f) / 2
        return np.exp(-(octahedron_distance - octahedron_center)**2 / (2 * (octahedron_width/2)**2))
    else:
        return 0.0

def create_cube_watermark(rows, cols, pose, mean_dis, min_f, max_f):

    T = pose[:3, 3]
    Rz = -pose[:3, 2]
    z_projection_length = np.dot(T, Rz)
    dis = abs(z_projection_length)
    
    min_f_scaled = min_f / mean_dis * dis
    max_f_scaled = max_f / mean_dis * dis
    

    view_direction = -pose[:3, 2]
    view_direction = view_direction / np.linalg.norm(view_direction)
    

    if abs(view_direction[0]) < 0.9:
        temp_vector = np.array([1, 0, 0])
    else:
        temp_vector = np.array([0, 1, 0])
    
    u_vector = np.cross(view_direction, temp_vector)
    u_vector = u_vector / np.linalg.norm(u_vector)
    v_vector = np.cross(view_direction, u_vector)
    v_vector = v_vector / np.linalg.norm(v_vector)
    

    mask = np.zeros((rows, cols), dtype=np.float32)
    

    diagonal = np.sqrt(rows**2 + cols**2)
    max_extent = diagonal / 2
    

    for i in range(rows):
        for j in range(cols):

            u_coord = (j - cols // 2) * max_extent / max(rows, cols) * 2
            v_coord = (i - rows // 2) * max_extent / max(rows, cols) * 2
            

            point_3d = u_coord * u_vector + v_coord * v_vector
            

            mask[i, j] = watermark_function_cube_shell_smooth(point_3d, min_f_scaled, max_f_scaled)
    
    return mask

def create_tetrahedron_watermark(rows, cols, pose, mean_dis, min_f, max_f):

    T = pose[:3, 3]
    Rz = -pose[:3, 2]
    z_projection_length = np.dot(T, Rz)
    dis = abs(z_projection_length)
    
    min_f_scaled = min_f / mean_dis * dis
    max_f_scaled = max_f / mean_dis * dis
    

    view_direction = -pose[:3, 2]
    view_direction = view_direction / np.linalg.norm(view_direction)
    

    if abs(view_direction[0]) < 0.9:
        temp_vector = np.array([1, 0, 0])
    else:
        temp_vector = np.array([0, 1, 0])
    
    u_vector = np.cross(view_direction, temp_vector)
    u_vector = u_vector / np.linalg.norm(u_vector)
    v_vector = np.cross(view_direction, u_vector)
    v_vector = v_vector / np.linalg.norm(v_vector)
    

    mask = np.zeros((rows, cols), dtype=np.float32)
    

    diagonal = np.sqrt(rows**2 + cols**2)
    max_extent = diagonal / 2
    

    for i in range(rows):
        for j in range(cols):

            u_coord = (j - cols // 2) * max_extent / max(rows, cols) * 2
            v_coord = (i - rows // 2) * max_extent / max(rows, cols) * 2
            

            point_3d = u_coord * u_vector + v_coord * v_vector
            

            mask[i, j] = watermark_function_tetrahedron_shell_smooth(point_3d, min_f_scaled, max_f_scaled)
    
    return mask

def create_octahedron_watermark(rows, cols, pose, mean_dis, min_f, max_f):

    T = pose[:3, 3]
    Rz = -pose[:3, 2]
    z_projection_length = np.dot(T, Rz)
    dis = abs(z_projection_length)
    
    min_f_scaled = min_f / mean_dis * dis
    max_f_scaled = max_f / mean_dis * dis
    

    view_direction = -pose[:3, 2]
    view_direction = view_direction / np.linalg.norm(view_direction)
    

    if abs(view_direction[0]) < 0.9:
        temp_vector = np.array([1, 0, 0])
    else:
        temp_vector = np.array([0, 1, 0])
    
    u_vector = np.cross(view_direction, temp_vector)
    u_vector = u_vector / np.linalg.norm(u_vector)
    v_vector = np.cross(view_direction, u_vector)
    v_vector = v_vector / np.linalg.norm(v_vector)
    

    mask = np.zeros((rows, cols), dtype=np.float32)
    

    diagonal = np.sqrt(rows**2 + cols**2)
    max_extent = diagonal / 2
    

    for i in range(rows):
        for j in range(cols):

            u_coord = (j - cols // 2) * max_extent / max(rows, cols) * 2
            v_coord = (i - rows // 2) * max_extent / max(rows, cols) * 2
            

            point_3d = u_coord * u_vector + v_coord * v_vector
            

            mask[i, j] = watermark_function_octahedron_shell_smooth(point_3d, min_f_scaled, max_f_scaled)
    
    return mask

def create_ring_watermark(rows, cols, pose, mean_dis, min_f, max_f):
    mask = np.zeros((rows, cols), np.float32)
    crow, ccol = rows // 2, cols // 2
    

    T = pose[:3, 3]
    Rz = -pose[:3, 2]
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

def freq_modify_with_pose(image, pose, mean_dis, args):
    rows, cols, c = image.shape
    

    if args.shape == 'ring':
        mask = create_ring_watermark(rows, cols, pose, mean_dis, args.min_f, args.max_f)
        print(f"Using ring watermark with frequency range: {args.min_f} - {args.max_f}")
    elif args.shape == 'cube':
        mask = create_cube_watermark(rows, cols, pose, mean_dis, args.min_f, args.max_f)
        print(f"Using cube watermark with frequency range: {args.min_f} - {args.max_f}")
    elif args.shape == 'tetrahedron':
        mask = create_tetrahedron_watermark(rows, cols, pose, mean_dis, args.min_f, args.max_f)
        print(f"Using tetrahedron watermark with frequency range: {args.min_f} - {args.max_f}")
    elif args.shape == 'octahedron':
        mask = create_octahedron_watermark(rows, cols, pose, mean_dis, args.min_f, args.max_f)
        print(f"Using octahedron watermark with frequency range: {args.min_f} - {args.max_f}")
    else:
        raise ValueError(f"Unsupported shape: {args.shape}. Use 'ring', 'cube', 'tetrahedron' or 'octahedron'.")
    

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

def log(self, *args, **kwargs):
    if self.local_rank == 0:
        if not self.mute: 
            #print(*args)
            self.console.print(*args, **kwargs)
        if self.log_ptr: 
            print(*args, file=self.log_ptr)
            self.log_ptr.flush() # write immediately to file

def calculate_average_z_projection_length(data):
    z_projection_lengths = []
    
    for frame in data['frames']:
        pose = frame['transform_matrix']
        
        # Convert the transform matrix to a numpy array for easier manipulation
        pose_matrix = np.array(pose)
        
        # Extract the translation vector T
        T = pose_matrix[:3, 3]
        
        # Extract the third column of the rotation matrix (camera's z-axis direction)

        Rz = -pose_matrix[:3, 2]
        
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

    print(f"Using watermark shape: {args.shape}")
    print(f"Frequency range: {args.min_f} - {args.max_f}")
    print(f"Watermark strength: {args.watermark_strength}")

    # for name in ["train"]:
    for name in ["train", "val", "test"]:
        pose_path = os.path.join(output_data_path, f"transforms_{name}.json")
    
        with open(pose_path, 'r') as f:
            data = json.load(f)
        if name == "train":
            mean_dis = calculate_average_z_projection_length(data)
            print(f"Calculated mean distance: {mean_dis:.4f}")
            
        for frame in tqdm(data['frames'], desc=f"Processing {name} frames with {args.shape} watermark"):
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
            "shape": args.shape,
            "watermark_strength": args.watermark_strength,
            "coordinate_system": "Camera forward direction: -Z axis (X: right, Y: up, -Z: forward)"
    }
    with open(os.path.join(output_data_path, f"setting.json"), 'w') as file:
        json.dump(data, file, indent=4)
#     shutil.copyfile(os.path.join(output_data_path, f"setting.json"), os.path.join(input_data_path, f"setting.json"))

if __name__ == '__main__':
    args = config_parser()
    w_generator(args)