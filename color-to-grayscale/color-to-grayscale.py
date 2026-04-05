def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    # Write code here
    h, w = len(image), len(image[0])
    ans = [[0]*w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            ans[i][j] = 0.299 * image[i][j][0] + 0.587 * image[i][j][1] + 0.114 * image[i][j][2]

    return ans 