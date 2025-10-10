import numpy as np
import logging
from cupyx.scipy.ndimage import map_coordinates

logger = logging.getLogger('job_executor')

class GPU_Convert:
    def __init__(self, image_shape):
        h, w , _ = image_shape
        SQUARE_SIDE  = h//2
        self.cp = __import__('cupy')
        self.cupyx_ndimage = __import__('cupyx.scipy.ndimage')
        self.coors_xy = self.uv2coor(self.xyz2uv(self.xyzcube(SQUARE_SIDE)),h, w)
        logger.info(f"Running inference on GPU")

    def xyzcube(self, face_w):
        '''
        Return the xyz coordinates of the unit cube in [F R B L U D] format.
        '''
        out = self.cp.zeros((face_w, face_w * 6, 3), dtype=self.cp.float32)
        rng = self.cp.linspace(-0.5, 0.5, num=face_w, dtype=self.cp.float32)
        grid = self.cp.stack(self.cp.meshgrid(rng, -rng), -1)

        # Front face (z = 0.5)
        out[:, 0*face_w:1*face_w, [0, 1]] = grid
        out[:, 0*face_w:1*face_w, 2] = 0.5

        # Right face (x = 0.5)
        grid_r = self.cp.flip(grid, axis=1)
        out[:, 1*face_w:2*face_w, [2, 1]] = grid_r
        out[:, 1*face_w:2*face_w, 0] = 0.5

        # Back face (z = -0.5)
        grid_b = self.cp.flip(grid, axis=1)
        out[:, 2*face_w:3*face_w, [0, 1]] = grid_b
        out[:, 2*face_w:3*face_w, 2] = -0.5

        # Left face (x = -0.5)
        out[:, 3*face_w:4*face_w, [2, 1]] = grid
        out[:, 3*face_w:4*face_w, 0] = -0.5

        # Up face (y = 0.5)
        grid_u = self.cp.flip(grid, axis=0)  
        out[:, 4*face_w:5*face_w, [0, 2]] = grid_u
        out[:, 4*face_w:5*face_w, 1] = 0.5

        # Down face (y = -0.5)
        out[:, 5*face_w:6*face_w, [0, 2]] = grid
        out[:, 5*face_w:6*face_w, 1] = -0.5

        return out
    
    def xyz2uv(self, xyz):
        '''
        xyz: cp.ndarray in shape of [..., 3]
        '''
        x, y, z = self.cp.split(xyz, 3, axis=-1)
        u = self.cp.arctan2(x, z)
        c = self.cp.sqrt(x**2 + z**2)
        v = self.cp.arctan2(y, c)

        return self.cp.concatenate([u, v], axis=-1)

    def uv2coor(self, uv, h, w):
        '''
        uv: cp.ndarray in shape of [..., 2]
        h: int, height of the equirectangular image
        w: int, width of the equirectangular image
        '''
        u, v = self.cp.split(uv, 2, axis=-1)
        coor_x = (u / (2 * self.cp.pi) + 0.5) * w - 0.5
        coor_y = (-v / self.cp.pi + 0.5) * h - 0.5

        return np.concatenate([coor_x, coor_y], axis=-1)
    
    def sample_equirec(self, e_img, coor_xy, order):
        w = e_img.shape[1]
        coor_x, coor_y = self.cp.split(coor_xy, 2, axis=-1)
        pad_u = self.cp.roll(e_img[[0]], w // 2, axis=1)
        pad_d = self.cp.roll(e_img[[-1]], w // 2, axis=1)
        e_img = self.cp.concatenate([e_img, pad_d, pad_u], axis=0)
        result = map_coordinates(e_img, self.cp.array([coor_y, coor_x]), order=order, mode='wrap')[..., 0]
        return result

    def e2c(self, e_img, coor_xy):
        c = e_img.shape[2]
        e_img = self.cp.asarray(e_img)  # Convert e_img to CuPy array if it's not already
        coor_xy = self.cp.asarray(coor_xy)  # Convert coor_xy to CuPy array if it's not already
        cubemap = self.cp.stack([self.sample_equirec(e_img[..., i], coor_xy, order=1) for i in range(c)], axis=-1)
        cubemap_faces = self.cp.array_split(cubemap, 6, axis=1)
        cubemap_dict = {k: self.cp.asnumpy(cubemap_faces[i]) for i, k in enumerate(['F', 'R', 'B', 'L', 'U', 'D'])}
        
        return cubemap_dict

    def convert_to_cubemaps(self, img):
        cubemap_images_dict = self.e2c(img, self.coors_xy)
        return cubemap_images_dict

def equirectangular_to_dicemap(image_np):
    
    converter = GPU_Convert(image_np.shape)
    cubemap_images = converter.convert_to_cubemaps(image_np)

    DICEMAP_FACE_LAYOUT = {  # (Column, Row)
        'R': (2, 1),  # Right
        'L': (0, 1),  # Left
        'U': (1, 0),  # Up
        'D': (1, 2),  # Down
        'F': (1, 1),  # Front
        'B': (3, 1),  # Back
    }

    sample_face = cubemap_images['F']
    face_h, face_w = sample_face.shape[:2]
    num_channels = sample_face.shape[2] if sample_face.ndim == 3 else 1
    dtype = sample_face.dtype
    
    assert face_h == face_w, "Cubemap faces must be square"

    # Create the new dicemap canvas
    dicemap_h = 3 * face_h
    dicemap_w = 4 * face_w


    dicemap_shape = (dicemap_h, dicemap_w, num_channels)
    dicemap = np.zeros(dicemap_shape, dtype=dtype)

    for face, img in cubemap_images.items():
        (grid_x, grid_y) = DICEMAP_FACE_LAYOUT[face]

        paste_x = grid_x * face_w
        paste_y = grid_y * face_h

        # Define the destination slice in the large array
        y_slice = slice(paste_y, paste_y + face_h)
        x_slice = slice(paste_x, paste_x + face_w)

        # Assign the face image to the slice
        dicemap[y_slice, x_slice] = img

    return dicemap