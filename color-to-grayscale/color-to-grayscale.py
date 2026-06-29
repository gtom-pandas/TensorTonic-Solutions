def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    
    Parameters:
    image (list of lists): 3D list representing an RGB image (H × W × 3)
    
    Returns:
    list of lists: 2D list representing the grayscale image (H × W)
    """
    # Initialize the grayscale image
    grayscale_image = []
    
    # Iterate over each row
    for row in image:
        grayscale_row = []
        # Iterate over each pixel in the row
        for pixel in row:
            R, G, B = pixel
            # Compute the grayscale intensity using luminance formula
            Y = 0.299 * R + 0.587 * G + 0.114 * B
            grayscale_row.append(Y)
        grayscale_image.append(grayscale_row)
    
    return grayscale_image