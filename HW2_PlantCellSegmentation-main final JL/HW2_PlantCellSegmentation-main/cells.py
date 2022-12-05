import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import skimage.io
from skimage.transform import resize
from skimage.color import rgb2gray
import time
#from unionfind import unionfind

class UnionFind:
    def __init__(self, N):
        self._parent = list(range(N))
    
    def root(self, i):
        """
        Follow parent pointers until reaching a root
        Parameters
        ----------
        i: int
            The starting node 
        
        Returns
        -------
        The root node of i
        """
        while self._parent[i] != i:
            i = self._parent[i]
        return i
    
    def find(self, i, j):
        """
        Return true if i and j are in the same component, or
        false otherwise
        Parameters
        ----------
        i: int
            Index of first element
        j: int
            Index of second element
        """
        return self.root(i) == self.root(j)
    
    def union(self, i, j):
        """
        Merge the two sets containing i and j, or do nothing if they're
        in the same set
        Parameters
        ----------
        i: int
            Index of first element
        j: int
            Index of second element
        """
        root_i = self.root(i)
        root_j = self.root(j)
        if root_i != root_j:
            self._parent[root_j] = root_i

def load_cells_grayscale(filename, n_pixels = 0):
    """
    Load in a grayscale image of the cells, where 1 is maximum brightness
    and 0 is minimum brightness

    Parameters
    ----------
    filename: string
        Path to image holding the cells
    n_pixels: int
        Number of pixels in the image
    
    Returns
    -------
    ndarray(N, N)
        A square grayscale image
    """
    cells_original = skimage.io.imread(filename)
    cells_gray = rgb2gray(cells_original)
    # Denoise a bit with a uniform filter
    cells_gray = ndimage.uniform_filter(cells_gray, size=10)
    cells_gray = cells_gray - np.min(cells_gray)
    cells_gray = cells_gray/np.max(cells_gray)
    N = int(np.sqrt(n_pixels))
    if n_pixels > 0:
        # Resize to a square image
        cells_gray = resize(cells_gray, (N, N), anti_aliasing=True)
    return cells_gray


def permute_labels(labels):
    """
    Shuffle around labels by raising them to a prime and
    modding by a large-ish prime, so that cells are easier
    to see against their backround
    Parameters
    ----------
    labels: ndarray(M, N)
        An array of labels for the pixels in the image
    Returns
    -------
    labels_shuffled: ndarray(M, N)
        A new image where the labels are different but still
        the same within connected components
    """
    return (labels**31) % 833


## TODO: Fill in your code here
def get_cell_labels(image, threshold):
    width, height = image.shape  # shortcut to extract shape of image
    list1 = np.zeros((width, height))
    #print(UnionFind, width * height)
    union = UnionFind(width * height)

    # [[0, 1, 0]  0
    #  [0, 1, 0]  1
    #  [0, 1*, 1]] 2

    # Formula: len(row)*i + j
    # 3 * 2 + 1 = 7

    # [0, 1, 2, 3, 4, 5, 6. 7, 8]

    for i in range(width):
        for j in range(height):
            # checks if horizontal neighbor is also above threshold
            if i + 1 < width:
                if image[i, j] > threshold and image[i+1, j] > threshold:
                    union.union(union_func(i, j, width), union_func(i+1, j, width))

            # checks if vertical neighbor is also above threshold
            if j + 1 < width:
                if image[i, j] > threshold and image[i, j+1] > threshold:
                    union.union(union_func(i, j, width), union_func(i, j+1, width))
    
    for i in range(width):
        for j in range(height):
            # Set list1[i, j] = ?, where ? is an integer representing a label
            list1[i, j] = union.root(union_func(i, j, width))
            
            
    return list1

def union_func(i, j, width):
    return width * i + j

def get_cluster_centers(list1):
    #these code comments are more so for my own notes and thought process
    # create empty list
    # pull data from list1 into list of lists
    width, height = list1.shape  # shortcut to extract shape of image

    average_centers = [];
    #list2empty = []  # append lists to this, might be easier to use a dictionary
                # where key is label
    list2empty = [[] for z in range((len(labels))**2)]
    

    
    # for every label, add every pixel in that label to a list
    # add that list to list2
    for i in range(width):
        for j in range(height):
            list2empty[int(list1[i][j])].append([i,j])
    
    for j in range(len(list2empty)):
        if (len(list2empty[j])) > 1:
            #counter = 0
            totalx = 0
            totaly = 0 
            for n in range(len(list2empty[j])):
                #print(list2empty[j][n])
                #gets the index so we can do the average math
                totalx += list2empty[j][n][0]
                totaly += list2empty[j][n][1]
            avgx = totalx/(len(list2empty[j]))
            avgy = totaly/(len(list2empty[j]))
            average_centers.append([avgx, avgy])
            
    return average_centers
                
            
        
        
            
            
            
            
            
        
            
            
            # First condition
            # list1[i, j] (label) assigned to list if never seen before
            # append list1[i, j] to that new list


            # Second condition
            # list1[i, j] is a label we've seen before
            # Get that list, and append pixel (i,j) to it
            # list2[label] is a list of all pixels with that label
            # list2[label].append((i,j))

            #pass
    #print(list2empty)


if __name__ == '__main__':
    thresh = 0.8
    I = load_cells_grayscale("Cells.jpg")
    labels = get_cell_labels(I, thresh)
    plt.imshow(permute_labels(labels))
    plt.savefig("labels.png", bbox_inches='tight')
    plt.show()
    x = get_cluster_centers(labels)
    x = np.array(x)
    cells_original = plt.imread("Cells.jpg")
    plt.imshow(cells_original)
    plt.scatter(x[:, 1], x[:, 0], c='C2')#shows plot for testing 
    plt.show()
   