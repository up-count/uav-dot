from typing import Tuple, List, Optional

import cv2
import numpy as np



class MaskGeneratorV2:
    def __init__(self, num_classes: int, image_shape: Tuple[int, int], mask_shape: Tuple[int, int]):
        self._num_classes = num_classes
        self._image_shape = image_shape
        self._mask_shape = mask_shape
        
        self._scale = (self._mask_shape[0] / self._image_shape[0], self._mask_shape[1] / self._image_shape[1])
    
    def generate_mask(self, keypoints: List[Tuple[int, int]], radius: int = 1):
        label = np.zeros((self._num_classes, self._mask_shape[1], self._mask_shape[0]), dtype=np.float32)
                        
        for key in keypoints:
            x = int(key[1] * self._scale[0])
            y = int(key[2] * self._scale[1])
            
            self.draw_umich_gaussian(label[int(key[0])], (x, y), radius=radius)
 
        return np.array(label)
    
    @staticmethod
    def gaussian2D(shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m+1,-n:n+1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def draw_umich_gaussian(self, heatmap, center, radius, k=1):
        assert type(radius) == int, 'radius must be an int'
        
        diameter = 2 * radius + 1
        gaussian = self.gaussian2D((diameter, diameter), sigma=diameter / 6)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap
