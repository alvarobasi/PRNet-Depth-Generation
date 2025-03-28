import numpy as np
import os
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp
from time import time
from PIL import Image
import cv2

from predictor import PosPrediction


class PRN:
    ''' Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network
    Args:
        is_dlib(bool, optional): If true, dlib is used for detecting faces.
        is_opencv(bool, optional): If true, opencv is used for extracting texture.
        prefix(str, optional): If run at another folder, the absolute path is needed to load the data.
    '''
    def __init__(self, is_dlib = False, is_opencv = False, prefix = '.'):

        # resolution of input and output image size.
        self.resolution_inp = 256
        self.resolution_op = 256

        #---- load detectors
        # if is_dlib:
        #     import dlib
        #     detector_path = os.path.join(prefix, 'Data/net-data/mmod_human_face_detector.dat')
        #     self.face_detector = dlib.cnn_face_detection_model_v1(
        #             detector_path)

        # if is_opencv:
        #     import cv2

        #---- load PRN 
        self.pos_predictor = PosPrediction(self.resolution_inp, self.resolution_op)
        prn_path = os.path.join(prefix, 'Data/net-data/256_256_resfcn256_weight')
        if not os.path.isfile(prn_path + '.data-00000-of-00001'):
            print("please download PRN trained model first.")
            exit()
        self.pos_predictor.restore(prn_path)

        # uv file
        self.uv_kpt_ind = np.loadtxt(prefix + '/Data/uv-data/uv_kpt_ind.txt').astype(np.int32) # 2 x 68 get kpt
        self.face_ind = np.loadtxt(prefix + '/Data/uv-data/face_ind.txt').astype(np.int32) # get valid vertices in the pos map
        self.triangles = np.loadtxt(prefix + '/Data/uv-data/triangles.txt').astype(np.int32) # ntri x 3

    def dlib_detect(self, image):
        return self.face_detector(image, 1)

    def net_forward(self, image):
        ''' The core of out method: regress the position map of a given image.
        Args:
            image: (256,256,3) array. value range: 0~1
        Returns:
            pos: the 3D position map. (256, 256, 3) array.
        '''
        return self.pos_predictor.predict(image)

    def process(self, input, image_info = None, FaceRect_name_full = None, image_shape = None):
        ''' process image with crop operation.
        Args:
            input: (h,w,3) array or str(image path). image value range:1~255. 
            image_info(optional): the bounding box information of faces. if None, will use dlib to detect face. 

        Returns:
            pos: the 3D position map. (256, 256, 3).
        '''
        if isinstance(input, str):
            try:
                image = imread(input)
            except IOError:
                print("error opening file: ", input)
                return None
        else:
            image = input

        if image.ndim < 3:
            image = np.tile(image[:,:,np.newaxis], [1,1,3])

        if image_info is not None:
            if np.max(image_info.shape) > 4: # key points to get bounding box
                kpt = image_info
                if kpt.shape[0] > 3:
                    kpt = kpt.T
                left = np.min(kpt[0, :]); right = np.max(kpt[0, :]); 
                top = np.min(kpt[1,:]); bottom = np.max(kpt[1,:])
            else:  # bounding box
                bbox = image_info
                left = bbox[0]; right = bbox[1]; top = bbox[2]; bottom = bbox[3]
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
            size = int(old_size*1.6)
            print(left, right, top, bottom)
        elif FaceRect_name_full is not None:
            fid = open(FaceRect_name_full, 'r')
            lines = fid.readlines()
            fid.close()
            floatlines = [float(x) for x in lines]
            left, top, w, h = floatlines[:4]
            #top, left, h, w = [int(float(x)) for x in lines]

            right = left + w
            bottom = top + h

            '''
            left = int(256.0 * left / image_shape[0])
            right = int(256.0 * right / image_shape[0])
            top = int(256.0 * top / image_shape[1])
            bottom = int(256.0 * bottom / image_shape[1])
            '''

            print(left, right, top, bottom)
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size*0.14])
            size = int(old_size*1.58)            
        else:
            detected_faces = self.dlib_detect(image)
            if len(detected_faces) == 0:
                print('warning: no detected face')
                return None

            d = detected_faces[0].rect ## only use the first detected face (assume that each input image only contains one face)
            left = d.left(); right = d.right(); top = d.top(); bottom = d.bottom()
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size*0.14])
            size = int(old_size*1.58)
            print(left, right, top, bottom)

        # crop image
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        DST_PTS = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        
        image = image/255.
        cropped_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))

        '''
        ## test cropped_image
        cropped_image = np.array(cropped_image*255.0, np.uint8)
        tmp_pil = Image.fromarray(cropped_image)
        tmp_pil.save('test_face_haven.jpg')
        exit(1)
        '''

        # run our net
        #st = time()
        cropped_pos = self.net_forward(cropped_image)
        #print 'net time:', time() - st

        # restore 
        cropped_vertices = np.reshape(cropped_pos, [-1, 3]).T
        z = cropped_vertices[2,:].copy()/tform.params[0,0]
        cropped_vertices[2,:] = 1
        vertices = np.dot(np.linalg.inv(tform.params), cropped_vertices)
        vertices = np.vstack((vertices[:2,:], z))
        pos = np.reshape(vertices.T, [self.resolution_op, self.resolution_op, 3])
        
        return pos
            
    def get_landmarks(self, pos):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            kpt: 68 3D landmarks. shape = (68, 3).
        '''
        kpt = pos[self.uv_kpt_ind[1,:], self.uv_kpt_ind[0,:], :]
        return kpt


    def get_vertices(self, pos):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            vertices: the vertices(point cloud). shape = (num of points, 3). n is about 40K here.
        '''
        all_vertices = np.reshape(pos, [self.resolution_op**2, -1]);
        vertices = all_vertices[self.face_ind, :]

        return vertices

    def get_texture(self, image, pos):
        ''' extract uv texture from image. opencv is needed here.
        Args:
            image: input image.
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            texture: the corresponding colors of vertices. shape = (num of points, 3). n is 45128 here.
        '''
        texture = cv2.remap(image, pos[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
        return texture


    def get_colors(self, image, vertices):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            colors: the corresponding colors of vertices. shape = (num of points, 3). n is 45128 here.
        '''
        [h, w, _] = image.shape
        vertices[:,0] = np.minimum(np.maximum(vertices[:,0], 0), w - 1)  # x
        vertices[:,1] = np.minimum(np.maximum(vertices[:,1], 0), h - 1)  # y
        ind = np.round(vertices).astype(np.int32)
        colors = image[ind[:,1], ind[:,0], :] # n x 3

        return colors







