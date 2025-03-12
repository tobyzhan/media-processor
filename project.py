"""
DSC 20 Project
Name(s): Toby Zhang
PID(s):  A18648052
Sources: None
"""

import numpy as np
import os
from PIL import Image
import wave
import struct
# import matplotlib.pyplot as plt

NUM_CHANNELS = 3


# --------------------------------------------------------------------------- #

# YOU SHOULD NOT MODIFY THESE TWO METHODS

def img_read_helper(path):
    """
    Creates an RGBImage object from the given image file
    """
    # Open the image in RGB
    img = Image.open(path).convert("RGB")
    # Convert to numpy array and then to a list
    matrix = np.array(img).tolist()
    # Use student's code to create an RGBImage object
    return RGBImage(matrix)


def img_save_helper(path, image):
    """
    Saves the given RGBImage instance to the given path
    """
    # Convert list to numpy array
    img_array = np.array(image.get_pixels())
    # Convert numpy array to PIL Image object
    img = Image.fromarray(img_array.astype(np.uint8))
    # Save the image object to path
    img.save(path)


# --------------------------------------------------------------------------- #

# Part 1: RGB Image #
class RGBImage:
    """
    Represents an image in RGB format
    """

    def __init__(self, pixels):
        """
        Initializes a new RGBImage object

        # Test with non-rectangular list
        >>> pixels = [
        ...              [[255, 255, 255], [255, 255, 255]],
        ...              [[255, 255, 255]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        # Test instance variables
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.pixels
        [[[255, 255, 255], [0, 0, 0]]]
        >>> img.num_rows
        1
        >>> img.num_cols
        2
        
        # Test if 3 ints are in each
        
        >>> pixels = [
        ...              [[255, 255], [0, 0, 0]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError
        
        # test if pixel is not in range
        
        >>> pixels = [
        ...              [[255, 255, 256], [0, 0, 0]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        ValueError
        
        # test if pixel is not int
        
        >>> pixels = [
        ...              [['255', 255, 255.0], [0, 0, 0]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        ValueError
        
        # test if each row is a list
        >>> pixels = [
        ...              ([255, 255, 255], [0, 0, 0])
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError
        
        #test if pixels is not empty
        >>> pixels = []
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError
        
        #test row length
        
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]],
        ...              [[255, 0, 0]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError
        
        
        """
        # YOUR CODE GOES HERE #
        # Raise exceptions here
        
        if not isinstance(pixels, list):
            raise TypeError()
        
        if not pixels:
            raise TypeError()
        
        if all([isinstance(pixel, int) and 0 <= pixel <= 255 
                for row in pixels for col in row for pixel in col]) != True:
            raise ValueError()
        
        row_len = len(pixels[0])
        
        if all([isinstance(row, list) and len(row) != 0 
                and len(row) == row_len for row in pixels]) != True:
            raise TypeError()
        
        if all([isinstance(pxls, list) and len(pxls) == 3 
                for row in pixels for pxls in row]) != True:
            raise TypeError()
        
        self.pixels = pixels
        self.num_rows = len(pixels)
        self.num_cols = len(pixels[0])

    def size(self):
        """
        Returns the size of the image in (rows, cols) format

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.size()
        (1, 2)
        
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]],
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.size()
        (2, 2)
        """
        
        # YOUR CODE GOES HERE #
        return (self.num_rows, self.num_cols)

    def get_pixels(self):
        """
        Returns a copy of the image pixel array

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_pixels = img.get_pixels()

        # Check if this is a deep copy
        >>> img_pixels                               # Check the values
        [[[255, 255, 255], [0, 0, 0]]]
        >>> id(pixels) != id(img_pixels)             # Check outer list
        True
        >>> id(pixels[0]) != id(img_pixels[0])       # Check row
        True
        >>> id(pixels[0][0]) != id(img_pixels[0][0]) # Check pixel
        True
        """
        # YOUR CODE GOES HERE #
        return [[[num for num in col]for col in row]for row in self.pixels]

    def copy(self):
        """
        Returns a copy of this RGBImage object

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_copy = img.copy()

        # Check that this is a new instance
        >>> id(img_copy) != id(img)
        True
        """
        # YOUR CODE GOES HERE #
        return self.get_pixels()

    def get_pixel(self, row, col):
        """
        Returns the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid index
        >>> img.get_pixel(1, 0)
        Traceback (most recent call last):
        ...
        ValueError

        # Run and check the returned value
        >>> img.get_pixel(0, 0)
        (255, 255, 255)
        
        #check negative
        >>> img.get_pixel(-1, 0)
        Traceback (most recent call last):
        ...
        ValueError
        
        #check not int
        >>> img.get_pixel('3', 0)
        Traceback (most recent call last):
        ...
        TypeError
        
        #check cols
        >>> img.get_pixel(0, 4)
        Traceback (most recent call last):
        ...
        ValueError
        """
        # YOUR CODE GOES HERE #
        
        if not isinstance(row, int):
            raise TypeError()
        
        if not isinstance(col, int):
            raise TypeError()
        
        size = self.size()
        
        if (((0 <= row < size[0]) and
            (0 <= col < size[1]))!= True):
            raise ValueError()
        
        return tuple(self.pixels[row][col])

    def set_pixel(self, row, col, new_color):
        """
        Sets the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid new_color tuple
        >>> img.set_pixel(0, 0, (256, 0, 0))
        Traceback (most recent call last):
        ...
        ValueError
        
        # Test with an invalid column
        >>> img.set_pixel(0, 4, (255, 0, 0))
        Traceback (most recent call last):
        ...
        ValueError
        
        # Test with an invalid row
        >>> img.set_pixel('0', 0, (255, 0, 0))
        Traceback (most recent call last):
        ...
        TypeError
        
        #test if new_color is not tuple
        >>> img.set_pixel(0, 0, [255, 0, 0])
        Traceback (most recent call last):
        ...
        TypeError
        

        # Check that the R/G/B value with negative is unchanged
        >>> img.set_pixel(0, 0, (-1, 0, 0))
        >>> img.pixels
        [[[255, 0, 0], [0, 0, 0]]]
        
        # Check that the R/G/B value with negative is unchanged
        >>> img.set_pixel(0, 1, (-1, 10, -20))
        >>> img.pixels
        [[[255, 0, 0], [0, 10, 0]]]
        
        """
        # YOUR CODE GOES HERE #
        if not isinstance(row, int):
            raise TypeError()
        
        if not isinstance(col, int):
            raise TypeError()
        
        size = self.size()
        
        if (((0 <= row < size[0]) and
            (0 <= col < size[1]))!= True):
            raise ValueError()
        
        if ((all([isinstance(n, int) for n in new_color])
            and len(new_color) == 3 and isinstance(new_color, tuple))
            != True):
            raise TypeError()
        
        if all([n <= 255 for n in new_color]) != True:
            raise ValueError()
        
        for i in range(len(new_color)):
            if new_color[i] >= 0:
                self.pixels[row][col][i] = new_color[i]
            

        


# Part 2: Image Processing Template Methods #
class ImageProcessingTemplate:
    """
    Contains assorted image processing methods
    Intended to be used as a parent class
    """

    def __init__(self):
        """
        Creates a new ImageProcessingTemplate object

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        # YOUR CODE GOES HERE #
        self.cost = 0


    def get_cost(self):
        """
        Returns the current total incurred cost

        # Check that the cost value is returned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost = 50 # Manually modify cost
        >>> img_proc.get_cost()
        50
        """
        # YOUR CODE GOES HERE #
        return self.cost

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check if this is returning a new RGBImage instance
        >>> img_proc = ImageProcessingTemplate()
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_negate = img_proc.negate(img)
        >>> id(img) != id(img_negate) # Check for new RGBImage instance
        True
        
        # The following is a description of how this test works
        # 1 Create a processor
        # 2/3 Read in the input and expected output
        # 4 Modify the input
        # 5 Compare the modified and expected
        # 6 Write the output to file
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()                            # 1
        >>> img = img_read_helper('img/test_image_32x32.png')                 # 2
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')  # 3
        >>> img_negate = img_proc.negate(img)                               # 4
        >>> img_negate.pixels == img_exp.pixels # Check negate output       # 5
        True
        >>> img_save_helper('img/out/test_image_32x32_negate.png', img_negate)# 6
        """
        # YOUR CODE GOES HERE #
        negate = [[[255 - pxl for pxl in col] for col in row]for row in image.get_pixels()]
        
        return RGBImage(negate)
        

    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_gray.png')
        >>> img_gray = img_proc.grayscale(img)
        >>> img_gray.pixels == img_exp.pixels # Check grayscale output
        True
        >>> img_save_helper('img/out/test_image_32x32_gray.png', img_gray)
        """
        # YOUR CODE GOES HERE #
        
        gray_pxl = [[[sum(col) // 3 for pxl in col] for col in row]for row in image.get_pixels()]
        
        return RGBImage(gray_pxl)

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_rotate.png')
        >>> img_rotate = img_proc.rotate_180(img)
        >>> img_rotate.pixels == img_exp.pixels # Check rotate_180 output
        True
        >>> img_save_helper('img/out/test_image_32x32_rotate.png', img_rotate)
        """
        # YOUR CODE GOES HERE #
        
        input_arr = image.get_pixels()
        
        output_arr = input_arr[::-1]
        
        reverse = [row[::-1]for row in output_arr]

        return RGBImage(reverse)

    def get_average_brightness(self, image):
        """
        Returns the average brightness for the given image

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.get_average_brightness(img)
        86
        """
        # YOUR CODE GOES HERE #
        size = image.size()
        num_pxls = size[0] * size[1] 
        
        pxl_avgs = [[sum(col) // 3 for col in row]for row in image.get_pixels()]
        col_avgs = [sum(row) for row in pxl_avgs]
        row_avg = sum(col_avgs)
        
        return row_avg // num_pxls

        
    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_adjusted.png')
        >>> img_adjust = img_proc.adjust_brightness(img, 1.2)
        >>> img_adjust.pixels == img_exp.pixels # Check adjust_brightness
        True
        >>> img_save_helper('img/out/test_image_32x32_adjusted.png', img_adjust)
        
        
        #check wrong input for intensity
        >>> img_adjust2 = img_proc.adjust_brightness(img, 3)
        Traceback (most recent call last):
        ...
        TypeError
        """
        # YOUR CODE GOES HERE #
        if not isinstance(intensity, float):
            raise TypeError()
        
        adjusted = [[[int(pxl * intensity) if int(pxl * intensity) <= 255 else 255 for pxl in col]for col in row]for row in image.get_pixels()]

        return RGBImage(adjusted)
    


# Part 3: Standard Image Processing Methods #
class StandardImageProcessing(ImageProcessingTemplate):
    """
    Represents a standard tier of an image processor
    """

    def __init__(self):
        """
        Creates a new StandardImageProcessing object

        # Check that the cost was assigned
        >>> img_proc = StandardImageProcessing()
        >>> img_proc.cost
        0
        """
        # YOUR CODE GOES HERE #
        self.cost = 0
        self.coupons = 0

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check the expected cost
        >>> img_proc = StandardImageProcessing()
        >>> img_in = img_read_helper('img/square_32x32.png')
        >>> negated = img_proc.negate(img_in)
        >>> img_proc.get_cost()
        5

        # Check that negate works the same as in the parent class
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')
        >>> img_negate = img_proc.negate(img)
        >>> img_negate.pixels == img_exp.pixels # Check negate output
        True
        """
        # YOUR CODE GOES HERE #
        
        if self.coupons > 0:
            self.coupons -= 1
            
        else:
            self.cost += 5
            
        return super().negate(image)

    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image
        >>> img_proc = StandardImageProcessing()
        >>> img_in = img_read_helper('img/square_32x32.png')
        >>> negated = img_proc.grayscale(img_in)
        >>> img_proc.get_cost()
        6
        
        """
        # YOUR CODE GOES HERE #
        
        if self.coupons > 0:
            self.coupons -= 1
            
        else:
            self.cost += 6
            
        return super().grayscale(image)

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image
        
        >>> img_proc = StandardImageProcessing()
        >>> img_in = img_read_helper('img/square_32x32.png')
        >>> negated = img_proc.rotate_180(img_in)
        >>> img_proc.get_cost()
        10
        """
        # YOUR CODE GOES HERE #
        if self.coupons > 0:
            self.coupons -= 1
            
        else:
            self.cost += 10
        return super().rotate_180(image)
        
        

    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level
        """
        # YOUR CODE GOES HERE #
        
        if self.coupons > 0:
            self.coupons -= 1
            
        else:
            self.cost += 1
        
        return super().adjust_brightness(image, intensity)
        
    def redeem_coupon(self, amount):
        """
        Makes the given number of methods calls free

        # Check that the cost does not change for a call to negate
        # when a coupon is redeemed
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.redeem_coupon(1)
        >>> img = img_proc.rotate_180(img)
        >>> img_proc.get_cost()
        0
        
        >>> img_proc.redeem_coupon(2)
        >>> img = img_proc.negate(img)
        >>> img_proc.get_cost()
        0
        
        >>> img = img_proc.grayscale(img)
        >>> img_proc.get_cost()
        0
        
        >>> img = img_proc.adjust_brightness(img, 1.2)
        >>> img_proc.get_cost()
        1
        
        """
        # YOUR CODE GOES HERE #
        if not isinstance(amount, int):
            raise TypeError()
        
        if amount <= 0:
            raise ValueError()
        
        self.coupons += amount
    
            
        
        


# Part 4: Premium Image Processing Methods #
class PremiumImageProcessing(ImageProcessingTemplate):
    """
    Represents a paid tier of an image processor
    """

    def __init__(self):
        """
        Creates a new PremiumImageProcessing object

        # Check the expected cost
        >>> img_proc = PremiumImageProcessing()
        >>> img_proc.get_cost()
        50
        """
        # YOUR CODE GOES HERE #
        self.cost = 50

    def pixelate(self, image, block_dim):
        """
        Returns a pixelated version of the image, where block_dim is the size of 
        the square blocks.

        >>> img_proc = PremiumImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_pixelate = img_proc.pixelate(img, 4)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_pixelate.png')
        >>> img_exp.pixels == img_pixelate.pixels # Check pixelate output
        True
        >>> img_save_helper('img/out/test_image_32x32_pixelate.png', img_pixelate)
        """
        # YOUR CODE GOES HERE #
        
        output = image.get_pixels()
        
        for r in range(0, len(output), block_dim):
            
            for c in range(0, len(output[0]), block_dim):
                
                avg = [0,0,0]
                
                for block_r in range(r, min(r + block_dim, len(output))):
                    for block_c in range(c, min(c + block_dim, len(output[0]))):
                        avg[0] += output[block_r][block_c][0]
                        avg[1] += output[block_r][block_c][1]
                        avg[2] += output[block_r][block_c][2]
                
                
                        
                new_pixels = [x // (((min(r + block_dim, len(output))) - r) *  
                              ((min(c + block_dim, len(output[0]))) - c))
                                for x in avg]
                
                        
                      
                for block_r in range(r, min(r + block_dim, len(output))):
                    for block_c in range(c, min(c + block_dim, len(output[0]))):
                        output[block_r][block_c][0] = new_pixels[0]
                        output[block_r][block_c][1] = new_pixels[1]
                        output[block_r][block_c][2] = new_pixels[2]
                        
                            
        return RGBImage(output)

    def edge_highlight(self, image):
        """
        Returns a new image with the edges highlighted

        >>> img_proc = PremiumImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_edge = img_proc.edge_highlight(img)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_edge.png')
        >>> img_exp.pixels == img_edge.pixels # Check edge_highlight output
        True
        >>> img_save_helper('img/out/test_image_32x32_edge.png', img_edge)
        """
        # YOUR CODE GOES HERE #

        output = image.get_pixels()
        
        avgs = [[sum(col) // 3 for col in row ]for row in output]
        
        for r in range(0, len(avgs)):
            for c in range(0, len(avgs[r])):
                
                masked_val = avgs[r][c] * 8 
                
                top_l = (r - 1, c - 1)
                top_m = (r - 1, c)
                top_r = (r - 1, c + 1)
                
                mid_l = (r, c - 1)
                mid_r = (r, c + 1)
                
                bot_l = (r + 1, c - 1)
                bot_m = (r + 1, c)
                bot_r = (r + 1, c + 1)
                
                neighbors = [top_l, top_m, top_r, mid_l, mid_r, bot_l, bot_m, bot_r]
                
                for n in neighbors:
                    
                    if n[0] >= 0 and n[0] < len(avgs) and n[1] >= 0 and n[1] < len(avgs[r]):
                        
                        masked_val += avgs[n[0]][n[1]] * -1
                
                if masked_val >= 255:
                    masked_val = 255
                
                if masked_val <= 0:
                    masked_val = 0

                
                output[r][c] = [masked_val, masked_val, masked_val]
                
        return RGBImage(output)

# --------------------------------------------------------------------------- #

# YOU SHOULD NOT MODIFY THESE THREE METHODS

def audio_read_helper(path, visualize=False):
    """
    Creates an AudioWave object from the given WAV file
    """
    with wave.open(path, "rb") as wav_file:
        num_frames = wav_file.getnframes()  # Total number of frames
        num_channels = wav_file.getnchannels()  # Number of channels (1 for mono, 2 for stereo)
        sample_width = wav_file.getsampwidth()  # Number of bytes per sample (e.g., 2 for 16-bit)
        
        # Read the frames as bytes
        raw_bytes = wav_file.readframes(num_frames)
        
        # Determine the format string for struct.unpack()
        fmt = f"{num_frames * num_channels}{'h' if sample_width == 2 else 'B'}"
        
        # Convert bytes to a list of integers
        audio_data = list(struct.unpack(fmt, raw_bytes))

    return AudioWave(audio_data)


def audio_save_helper(path, audio, sample_rate = 44100):
    """
    Saves the given AudioWave instance to the given path as a WAV file
    """
    sample_rate = 44100  # 44.1 kHz standard sample rate
    num_channels = 1  # Mono
    sample_width = 2  # 16-bit PCM

    # Convert list to bytes
    byte_data = struct.pack(f"{len(audio.wave)}h", *audio.wave)
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(byte_data)


def audio_visualizer(path, start=0, end=5):
    """
    Visualizes the given WAV file
    x-axis: time (sec)
    y-axis: wave amplitude
    ---
    Parameters: 
        path (str): path to the WAV file
        start (int): start timestamp in seconds, default 0
        end (int): end timestamp in seconds, default 5
    """
    with wave.open(path, "rb") as wav_file:
        sample_freq = wav_file.getframerate()  # Sample rate
        n_samples = wav_file.getnframes()  # Total number of samples
        duration = n_samples/sample_freq # Duration of audio, in seconds

        if any([type(param) != int for param in [start, end]]):
            raise TypeError("start and end should be integers.")
        if (start < 0) or (start > duration) or (end < 0) or (end > duration) or start >= end:
            raise ValueError(f"Invalid timestamp: start and end should be between 0 and {int(duration)}, and start < end.")
        
        num_frames = wav_file.getnframes()  # Total number of frames
        num_channels = wav_file.getnchannels()  # Number of channels (1 for mono, 2 for stereo)
        sample_width = wav_file.getsampwidth()  # Number of bytes per sample (e.g., 2 for 16-bit)
        
        # Extract audio wave as list
        raw_bytes = wav_file.readframes(num_frames)
        fmt = f"{num_frames * num_channels}{'h' if sample_width == 2 else 'B'}"
        audio_data = list(struct.unpack(fmt, raw_bytes))

        # Plot the audio wave
        time = np.linspace(start, end, num=(end - start)*sample_freq)
        audio_data = audio_data[start*sample_freq:end*sample_freq]
        plt.figure(figsize=(15, 5))
        plt.ylim([-32768, 32767])
        plt.plot(time, audio_data)
        plt.title(f'Audio Plot of {path} from {start}s to {end}s')
        plt.ylabel('sound wave')
        plt.xlabel('time (s)')
        plt.xlim(start, end)
        plt.show()


# --------------------------------------------------------------------------- #

# Part 5: Multimedia Processing
class AudioWave():
    """
        Represents audio through a 1-dimensional array of amplitudes
    """
    def __init__(self, amplitudes):
        self.wave = amplitudes

class PremiumPlusMultimediaProcessing(PremiumImageProcessing):
    """
        Represents the paid tier of multimedia processing
    """
    def __init__(self):
        """
        Creates a new PremiumPlusMultimediaProcessing object

        # Check the expected cost
        >>> multi_proc = PremiumPlusMultimediaProcessing()
        >>> multi_proc.get_cost()
        75
        """
        # YOUR CODE GOES HERE #
        self.cost = 75
    
    def reverse_song(self, audio):
        """
        Reverses the audio of the song.

        >>> multi_proc = PremiumPlusMultimediaProcessing()
        >>> audio = audio_read_helper('audio/one_summers_day.wav')
        >>> audio_reversed = multi_proc.reverse_song(audio)
        >>> audio_exp = audio_read_helper('audio/exp/one_summers_day_reversed.wav')
        >>> audio_exp.wave == audio_reversed.wave # Check reverse_song output
        True
        >>> audio_save_helper('audio/out/one_summers_day_reversed.wav', audio_reversed)
        """
        # YOUR CODE GOES HERE #
        if not isinstance(audio, AudioWave):
            raise TypeError()
     
        return AudioWave([aud for aud in audio.wave[::-1]])
        
    
    def slow_down(self, audio, factor):
        """
        Slows down the song by a certain factor.

        >>> multi_proc = PremiumPlusMultimediaProcessing()
        >>> audio = audio_read_helper('audio/one_summers_day.wav')
        >>> audio_slow = multi_proc.slow_down(audio, 2)
        >>> audio_exp = audio_read_helper('audio/exp/one_summers_day_slow.wav')
        >>> audio_exp.wave == audio_slow.wave # Check slow_down output
        True
        >>> audio_save_helper('audio/out/one_summers_day_slow.wav', audio_slow)
        """
        # YOUR CODE GOES HERE #
        if not isinstance(audio, AudioWave):
            raise TypeError()
        
        if not isinstance(factor, int):
            raise TypeError()
        
        if factor <= 0:
            raise ValueError()
        
        return AudioWave([aud for aud in audio.wave for fac in range(factor)])
    
    
    def speed_up(self, audio, factor):
        """
        Speeds up the song by a certain factor.

        >>> multi_proc = PremiumPlusMultimediaProcessing()
        >>> audio = audio_read_helper('audio/one_summers_day.wav')
        >>> audio_sped_up = multi_proc.speed_up(audio, 2)
        >>> audio_exp = audio_read_helper('audio/exp/one_summers_day_sped_up.wav')
        >>> audio_exp.wave == audio_sped_up.wave # Check speed_up output
        True
        >>> audio_save_helper('audio/out/one_summers_day_sped_up.wav', audio_sped_up)
        """
        # YOUR CODE GOES HERE #
        if not isinstance(audio, AudioWave):
            raise TypeError()
        
        if not isinstance(factor, int):
            raise TypeError()
        
        if factor <= 0:
            raise ValueError()
        
        return AudioWave([aud for aud in audio.wave[::factor]])

    def reverb(self, audio):
        """
        Adds a reverb/echo effect to the song.

        >>> multi_proc = PremiumPlusMultimediaProcessing()
        >>> audio = audio_read_helper('audio/one_summers_day.wav')
        >>> audio_reverb = multi_proc.reverb(audio)
        >>> audio_exp = audio_read_helper('audio/exp/one_summers_day_reverb.wav')
        >>> audio_exp.wave == audio_reverb.wave # Check reverb output
        True
        >>> audio_save_helper('audio/out/one_summers_day_reverb.wav', audio_reverb)
        """
        # YOUR CODE GOES HERE #
        if not isinstance(audio, AudioWave):
            raise TypeError()
        
        # list(map(lambda x, y: x[]))
        return AudioWave([min(max(round(sum(audio.wave[i-j] * (1 - 0.2 * j) for j in range(5))), -32768), 32767) if i >= 4 else audio.wave[i] for i in range(len(audio.wave))])
        
        
    def clip_song(self, audio, start, end):
        """
        Clips a song based on a specified start and end.
        
        >>> multi_proc = PremiumPlusMultimediaProcessing()
        >>> audio = audio_read_helper('audio/one_summers_day.wav')
        >>> audio_clipped = multi_proc.clip_song(audio, 30, 70)
        >>> audio_exp = audio_read_helper('audio/exp/one_summers_day_clipped.wav')
        >>> audio_exp.wave == audio_clipped.wave # Check clip_song output
        True
        >>> audio_save_helper('audio/out/one_summers_day_clipped.wav', audio_clipped)
        """
        # YOUR CODE GOES HERE #
        if not isinstance(audio, AudioWave):
            raise TypeError()
        
        if not isinstance(start, int):
            raise TypeError()  
        
        if not isinstance(end, int):
            raise TypeError()  
        
        if start < 0 or start > 100: 
            raise ValueError()
        
        if end < 0 or end > 100: 
            raise ValueError()
        
        
        idx = len(audio.wave)
        
        output = audio.wave
        
        start_idx = int((start/100) * idx)
        
        end_idx = int((end/100) * idx) + 1
        
        if end <= start:
            return AudioWave([])
        
        else:
            return AudioWave(output[start_idx:end_idx])
        
        


# Part 6: Image KNN Classifier #
class ImageKNNClassifier:
    """
    Represents a simple KNNClassifier
    """

    def __init__(self, k_neighbors):
        """
        Creates a new KNN classifier object
        """
        # YOUR CODE GOES HERE #
        self.k_neighbors = k_neighbors

    def fit(self, data):
        """
        Stores the given set of data and labels for later
        """
        # YOUR CODE GOES HERE #

        if len(data) < self.k_neighbors:
            raise ValueError()
        
        self.data = data
        
        
    def distance(self, image1, image2):
        """
        Returns the distance between the given images

        >>> img1 = img_read_helper('img/steve.png')
        >>> img2 = img_read_helper('img/knn_test_img.png')
        >>> knn = ImageKNNClassifier(3)
        >>> knn.distance(img1, img2)
        15946.312896716909
        """
        # YOUR CODE GOES HERE #
        
        if image1.size() != image2.size():
            raise ValueError()
        
        if not isinstance(image1, RGBImage):
            raise TypeError()
        
        if not isinstance(image2, RGBImage):
            raise TypeError()
        
        img1 = image1.get_pixels()
        img2 = image2.get_pixels()
        
        dist = sum([(img1[row][col][pxl] - img2[row][col][pxl]) ** 2 for row in range(len(img1)) for col in range(len(img1[0])) for pxl in range(len(img1[0][0]))])
        
        return dist ** 0.5
        
        

    def vote(self, candidates):
        """
        Returns the most frequent label in the given list

        >>> knn = ImageKNNClassifier(3)
        >>> knn.vote(['label1', 'label2', 'label2', 'label2', 'label1'])
        'label2'
        >>> knn.vote(['label1', 'label2', 'label2', 'label1'])
        'label1'
        """
        # YOUR CODE GOES HERE #
        dct = {}
        
        for c in candidates:
            if c not in dct:
                dct[c] = 1
            
            else:
                dct[c] += 1
        
        return max(dct, key=dct.get)

    def predict(self, image):
        """
        Predicts the label of the given image using the labels of
        the K closest neighbors to this image

        The test for this method is located in the knn_tests method below
        """
        # YOUR CODE GOES HERE #
        if not self.data:
            raise ValueError()
        
        distances = []


        for img, label in self.data: #list of (image, label)
            distances.append((self.distance(image, img), label))
            
        
        distances.sort(key=lambda x: x[0])
        
        closest = distances[:self.k_neighbors]
        
        labels = []
        
        for img, label in closest:
            labels.append(label)
        
        best_label = self.vote(labels)
        
        return best_label
        

def knn_tests(test_img_path):
    """
    Function to run knn tests

    >>> knn_tests('img/knn_test_img.png')
    'nighttime'
    """
    # Read all of the sub-folder names in the knn_data folder
    # These will be treated as labels
    path = 'knn_data'
    data = []
    for label in os.listdir(path):
        label_path = os.path.join(path, label)
        # Ignore non-folder items
        if not os.path.isdir(label_path):
            continue
        # Read in each image in the sub-folder
        for img_file in os.listdir(label_path):
            train_img_path = os.path.join(label_path, img_file)
            img = img_read_helper(train_img_path)
            # Add the image object and the label to the dataset
            data.append((img, label))

    # Create a KNN-classifier using the dataset
    knn = ImageKNNClassifier(5)

    # Train the classifier by providing the dataset
    knn.fit(data)

    # Create an RGBImage object of the tested image
    test_img = img_read_helper(test_img_path)

    # Return the KNN's prediction
    predicted_label = knn.predict(test_img)
    return predicted_label
