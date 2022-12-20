import cv2
import numpy as np
import matplotlib.pyplot as plt


def Normalized_Img(img, Truncating = False):
    Image = img.copy()
    Image = np.float64(Image)
    
    if Truncating == True:
        Image[Image < 0] = 0
        Image[Image > 255] = 255
        
    try:
        for i in range(3):    
            Image[:,:,i] = Image[:,:,i] - np.min(Image[:,:,i])
            Image[:,:,i] = Image[:,:,i]/np.max(Image[:,:,i])*255
            
    except:
        Image = Image - np.min(Image)
        Image = Image/np.max(Image)*255  
    return Image    

def myImageFilter(img0, h):
    img = img0.astype(np.float64).copy()
    filter_heights, filter_widths = h.shape
    padd_heights = (filter_heights - 1)//2
    padd_widths = (filter_widths - 1)//2
    
    try:
        padded_img = np.zeros((img.shape[0] + 2*padd_heights,
                               img.shape[1] + 2*padd_widths, img.shape[2]))
    except:
        padded_img = np.zeros((img.shape[0] + 2*padd_heights,
                               img.shape[1] + 2*padd_widths))
        
    padded_img[padd_heights:-padd_heights, padd_widths:-padd_widths] = img.copy()
    padded_img[:, 0:padd_widths] = padded_img[:, 2*padd_widths:padd_widths:-1].copy()
    padded_img[:, -padd_widths:] = padded_img[:, -padd_widths-1:-2*padd_widths-1:-1].copy()
    padded_img[0:padd_heights, :] = padded_img[2*padd_heights:padd_heights:-1, :].copy()
    padded_img[-padd_heights:, :] = padded_img[-padd_heights-1:-2*padd_heights-1:-1, :].copy()
    
    img1 = np.zeros(img.shape).astype(np.float64)
    for i in range(filter_heights):
        for j in range(filter_widths):
            img1 += padded_img[i:i+padded_img.shape[0]-2*padd_heights,
                               j:j+padded_img.shape[1]-2*padd_widths]*h[i, j]
    return img1

def Guassian_Filter(sigma):
    L = np.floor(3*sigma)    
    n = np.arange(-L, L+1, 1)
    G_X = np.exp(-pow(n,2)/(2*pow(sigma, 2)))
    G_X = G_X/sum(G_X)
    G_Y = G_X
    G_Y = np.copy(G_X)
    G_Y.shape = (len(n), 1)
    G_X.shape = (1,len(n))
    return G_Y@G_X

def Non_Max_Suppression(img, L):
    padded_img = np.zeros((img.shape[0] + 2*L, img.shape[1] + 2*L))
    padded_img[L:-L, L:-L] = img.copy()
    
    non_max_suppress = np.zeros(img.shape)
    for i in range(L, padded_img.shape[0]-L):
        for j in range(L, padded_img.shape[1]-L):
            if np.max(padded_img[i-L:i+L, j-L:j+L]) == padded_img[i, j]:
                non_max_suppress[i-L, j-L] = padded_img[i,j]
    return non_max_suppress

def Harris_Corner_Detection(img,
                            sigma1 = 1,
                            sigma2 = 1,
                            k = 0.01,
                            threshold = 8,
                            n = 2):
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    G1 = Guassian_Filter(sigma1)
    G_X = np.zeros(G1.shape)
    G_Y = np.zeros(G1.shape)
    G_X[0:-1, :] = np.diff(G1, axis = 0)
    G_Y[:, 0:-1] = np.diff(G1, axis = 1)
    I_x = myImageFilter(img_gray, G_X)
    I_y = myImageFilter(img_gray, G_Y)
    
    G2 = Guassian_Filter(sigma2)    
    S_x2 = myImageFilter(pow(I_x, 2), G2)
    S_y2 = myImageFilter(pow(I_y, 2), G2)
    S_xy = myImageFilter(np.multiply(I_x, I_y), G2)

    det = np.multiply(S_x2, S_y2) - pow(S_xy, 2)
    trace = S_x2 + S_y2
    R = det - k*pow(trace, 2)
    R[R < threshold] = np.min(R)
    
    Interest_Points = np.where(Normalized_Img(Non_Max_Suppression(R, 2), True) > 0)
    
    logic1 = np.logical_and(Interest_Points[0] > 3*n, Interest_Points[0] + 3*n + 1 < img.shape[0])
    logic2 = np.logical_and(Interest_Points[1] > 3*n, Interest_Points[1] + 3*n + 1 < img.shape[1])
    logic = np.logical_and(logic1, logic2)
    pruned_Interest_Points = (Interest_Points[0][logic], Interest_Points[1][logic])
    
    Interest_Hist = np.zeros((3*(2*n+1)**2, pruned_Interest_Points[0].shape[0]))
    for i in range(pruned_Interest_Points[0].shape[0]):
        points = img[pruned_Interest_Points[0][i]-n : pruned_Interest_Points[0][i]+n+1,
                      pruned_Interest_Points[1][i]-n : pruned_Interest_Points[1][i]+n+1,:].copy()
                       
        Interest_Hist[:,i] = np.reshape(points, (1, 3*(2*n+1)**2), order = 'F')[0]
        
    return (pruned_Interest_Points, Interest_Hist)    


img1 = cv2.imread('im01.jpg')
img2 = cv2.imread('im02.jpg')

Interest_Points_1, Interest_Hist_1 = Harris_Corner_Detection(img1, k = 0.1, n = 5, threshold = 1000)
Interest_Points_2, Interest_Hist_2 = Harris_Corner_Detection(img2, k = 0.1, n = 5, threshold = 1000)

threshold = 0.6

candidate_correspond_points12 = []

for i in range(Interest_Hist_1.shape[1]):
    temp = np.tile(Interest_Hist_1[:, i], (Interest_Hist_2.shape[1], 1)).T
    dist = np.sum(abs(pow(Interest_Hist_2 - temp, 2)), axis = 0)   
    
    p1 = np.where(dist == np.min(dist))[0].copy()
    if p1.shape[0] != 1:
        continue
    d1 = dist[p1[0]].copy()
    dist[p1] = np.max(dist)
    p2 = np.where(dist == np.min(dist))[0][0].copy()
    d2 = dist[p2].copy()
    
    if d1/d2 <= threshold:
        candidate_correspond_points12.append((i, p1[0]))
    
candidate_correspond_points21 = []

for i in range(Interest_Hist_2.shape[1]):
    temp = np.tile(Interest_Hist_2[:, i], (Interest_Hist_1.shape[1], 1)).T
    dist = np.sum(abs(pow(Interest_Hist_1 - temp, 2)), axis = 0)  
    
    p1 = np.where(dist == np.min(dist))[0].copy()
    if p1.shape[0] != 1:
        continue
    d1 = dist[p1[0]].copy()
    dist[p1] = np.max(dist)
    p2 = np.where(dist == np.min(dist))[0][0].copy()
    d2 = dist[p2].copy()
    
    if d1/d2 <= threshold:
        candidate_correspond_points21.append((p1[0], i))

correspond_points = list(set(candidate_correspond_points21).intersection(set(candidate_correspond_points12)))

IMG = np.zeros((img1.shape[0], img1.shape[1] + img2.shape[1], 3))
IMG[:,:img1.shape[1]] = img1.copy()
IMG[:,img1.shape[1]:] = img2.copy()

COLOR = np.random.randint(0, 255, (len(correspond_points), 3))
for i in range(len(correspond_points)):
    pt1 = (Interest_Points_1[1][correspond_points[i][0]],
           Interest_Points_1[0][correspond_points[i][0]])
    pt2 = (Interest_Points_2[1][correspond_points[i][1]] + img1.shape[1],
           Interest_Points_2[0][correspond_points[i][1]])
    cv2.line(IMG, pt2, pt1, (int(COLOR[i,0]), int(COLOR[i,1]), int(COLOR[i,2])), thickness = 2)
    IMG = cv2.circle(IMG, pt2, 15, (int(COLOR[i,0]), int(COLOR[i,1]), int(COLOR[i,2])), thickness = -1)
    IMG = cv2.circle(IMG, pt1, 15, (int(COLOR[i,0]), int(COLOR[i,1]), int(COLOR[i,2])), thickness = -1)
    
cv2.imwrite('Result.jpg', IMG)