import numpy as np
import random as rnd
from scipy.ndimage.measurements import label

BORDER = 3
LINE_COLOR = [0, 0, 255]
RED = [255, 0, 0]
GREEN = [0, 255, 0]
YELLOW = [255, 255, 0]
BLACK = [0, 0, 0]

RULE = lambda X: np.logical_and(np.logical_and(X[40] == 1, X[18] == 0), X[35] == 0)

def show_matches(rulesVec, image):
    index = 0
    matches = image['blacknwhite'].copy()
    
    for x in range(4, len(matches) - 4):
        for y in range(4, len(matches[x]) -  4):
            isOK = False
            box = image['boxs'][index].ravel()
            for rule in rulesVec:
                vecRule = rule[0]
                if isOK == False:
                    if (vecRule(box) == True and image['actual'][x - 4, y - 4] == True):
                        matches[x, y] = GREEN
                        isOK = True
                    elif (vecRule(box) == True and image['actual'][x - 4, y - 4] == False):
                        matches[x, y] = RED
                        isOK = True
                    elif (vecRule(box) == False and image['actual'][x - 4, y - 4] == True):
                        matches[x, y] = YELLOW
                        isOK = True
            index+=1
            
    return matches

def live_matches(grayImage, colorImage, limit, rules, coloration):
    dist = 4
    prediction = np.zeros((len(grayImage) - dist, len(grayImage[0]) - dist), dtype='uint8')
    matches = colorImage.copy()
    
    boolImage = grayImage>limit
    
    for x in range(len(boolImage) - dist*2):
        for y in range(len(boolImage[x]) - dist*2):
            box = boolImage[x:x+9, y:y+9]
            boxVec = box.ravel()
            if(len(boxVec)==81):
                for RULE in rules: 
                    if(RULE[0](boxVec) == True):
                        matches[x + dist, y + dist] = coloration
                        prediction[x, y] = 1
                
    numFingers = countFingers(prediction, 20)
    return matches, numFingers

def countFingers(image, minSize):
    numFingers = 0
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]])

    connectedComp, numComp = label(image, structure) 
    
    for i in range(1, numComp+1):
        size = sum(sum(connectedComp[:][:] == i))
        if size >= minSize:
            numFingers += 1
    
    return numFingers

def from_float_to_uint8(floatImges):
    uint8Images = []
    
    for i in range(len(floatImges)):  
        data = floatImges[i] / floatImges[i].max() #normalizes data in range 0 - 255
        data = 255 * data
        image = data.astype(np.uint8)
        uint8Images.append(image)
        
    return uint8Images

def invert_image(image):
    return ~image

def insert_image(image, insertion, dims):
    xStart, xStop, yStart, yStop = dims
    
    image[xStart:xStop, yStart:yStop] = insertion
    
    return image

def reduce_image(image, newX, newY, blackSquare):    
    image = image.copy()
    
    xLength = len(image)
    yLength = len(image[0])
    
    xStart = round((xLength-newX)/2)
    yStart = round((yLength-newY)/2)
    
    xStop = xStart + newX
    yStop = yStart + newY
    
    newImage = image[xStart:xStop, yStart:yStop]
    
    #Right, Up, Left, Down Lines
    image[xStart:xStop+BORDER,  yStop:yStop+BORDER]   = LINE_COLOR
    image[xStart-BORDER:xStart, yStart:yStop+BORDER]  = LINE_COLOR
    image[xStart-BORDER:xStop,  yStart-BORDER:yStart] = LINE_COLOR
    image[xStop:xStop+BORDER,   yStart-BORDER:yStop]  = LINE_COLOR
    
    if(blackSquare):
        image[0:75,  0:250] = [0, 0, 0]
        
    dimensions = [xStart, xStop, yStart, yStop]
    
    return newImage, image, dimensions

def color_square(image):
    image = image.copy()
    
    xMax = len(image)
    yMax = len(image[0])
    
    xStart = rnd.randrange(xMax)
    yStart = rnd.randrange(yMax)
    xStop = rnd.randrange(xStart, xMax)
    yStop = rnd.randrange(yStart, yMax)
    
    return image