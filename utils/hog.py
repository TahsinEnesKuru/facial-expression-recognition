import numpy as np
import cv2
from PIL import Image

class HOGDescriptor:
    def __init__(self, win_size=(48, 48), block_size=(16, 16), block_stride=(8, 8), cell_size=(8, 8), nbins=9):
        # Initialize HOG descriptor parameters
        self.win_size = win_size  # Size of the image window
        self.block_size = block_size  # Size of blocks
        self.block_stride = block_stride  # Block stride
        self.cell_size = cell_size  # Size of cells
        self.nbins = nbins  # Number of bins for histograms
        
        # Create the HOG descriptor and detector
        self.hog = cv2.HOGDescriptor(self.win_size, self.block_size, self.block_stride, self.cell_size, self.nbins)
    
    def compute(self, image):
        """
        Compute the HOG descriptor for a given image.
        
        Parameters:
            image: The image for which to compute the HOG descriptor. Can be in the format of OpenCV image or PIL image.
        
        Returns:
            The HOG descriptor for the image.
        """
        # Check if the image is a PIL image and convert it to an OpenCV image if necessary
        if isinstance(image, Image.Image):
            image = self.pil_to_opencv(image)
        
        # Convert image to grayscale as HOG needs a single channel image
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute the HOG features
        hog_features = self.hog.compute(image)
        
        # Return the computed HOG features
        return hog_features.flatten()  # Flatten to make it suitable for feeding into classifiers
    
    @staticmethod
    def pil_to_opencv(image):
        """
        Convert PIL image to OpenCV format.
        
        Parameters:
            image: PIL image to be converted.
        
        Returns:
            The converted OpenCV image.
        """
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# Example usage
if __name__ == "__main__":
    hog_descriptor = HOGDescriptor()
    
    # Load an image using OpenCV
    opencv_image = cv2.imread('1.jpg')
    
    # Load an image using PIL
    pil_image = Image.open('1.jpg')
    
    # Compute HOG features
    hog_features_opencv = hog_descriptor.compute(opencv_image)
    hog_features_pil = hog_descriptor.compute(pil_image)