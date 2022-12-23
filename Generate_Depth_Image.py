import numpy as np
import scipy.io as sio
from skimage.io import imread, imsave
import cv2
import os
import re

from api import PRN
import utils.depth_image as DepthImage

def get_list_of_files(folder):
    """Retreives the list of all the image paths.

    Args:
        folder (str): Path to the dataset root folder.

    Returns:
        List: List containing the relative path of all the images contained within the dataset.
    """    
    # Function to search folders and subfolders to obtain all png files
    files_in_folder = os.listdir(folder)
    output_files = []
    # Iterate over all the entries
    for entry in files_in_folder:
        # Create full path
        full_path = os.path.join(folder, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(full_path):
            output_files = output_files + get_list_of_files(full_path)
        elif not re.match(r'.*\.txt$', entry) and not re.match(r'.*\.dat$', entry):
            output_files.append(full_path)
    return output_files

def process_path(path):
    parent, image_file = os.path.split(path)
    bbx_file = os.path.splitext(image_file)[0]+".dat"
    depth_file_name = image_file.split("_frame")[0]+"_depth.jpg"
    video_name = parent.split("/")[-1]

    # if os.path.exists("/home/alvaro/data_mount/Downloads_temp_ubuntu/OULU-NPU/dataset/cdcn/depth_maps/test/"+video_name):
    #     return

    if not os.path.exists("/home/alvaro/data_mount/Downloads_temp_ubuntu/OULU-NPU/dataset/cdcn/depth_maps/dev/"+video_name):
        os.makedirs("/home/alvaro/data_mount/Downloads_temp_ubuntu/OULU-NPU/dataset/cdcn/depth_maps/dev/"+video_name)

    if os.path.isfile("/home/alvaro/data_mount/Downloads_temp_ubuntu/OULU-NPU/dataset/cdcn/depth_maps/dev/"+video_name+"/"+depth_file_name):
        return

    image = imread(path)
    image_shape = [image.shape[0], image.shape[1]]

    f=open(parent+"/"+bbx_file,'r')
    lines=f.readlines()
    y1,x1,w,h=[float(ele) for ele in lines[0].split(',')][:4]
    f.close()
    pos = prn.process(image, np.array([x1, x1 + w, y1, y1 + h]), None, image_shape)

    kpt = prn.get_landmarks(pos)

    # 3D vertices
    vertices = prn.get_vertices(pos)

    depth_scene_map = DepthImage.generate_depth_image(vertices, kpt, image.shape, isMedFilter=True)

    cv2.imwrite("/home/alvaro/data_mount/Downloads_temp_ubuntu/OULU-NPU/dataset/cdcn/depth_maps/dev/"+video_name+"/"+depth_file_name, depth_scene_map)
    
prn = PRN(is_dlib = False, is_opencv = True) 
img_paths = get_list_of_files("/home/alvaro/data_mount/Downloads_temp_ubuntu/OULU-NPU/dataset/cdcn/frames/dev")

if __name__ ==  '__main__':
    # pool = mp.Pool(mp.cpu_count() - 1)
    # pool.map(video_path, img_paths)
    # pool.close()
    # pool.join()
    for path in img_paths:
        process_path(path)