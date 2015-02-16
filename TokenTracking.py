#A simple game using token tracking using a camera

import numpy as np
import cv2
import pyttsx
import math

def rectify(h):
  h = h.reshape((4,2))
  hnew = np.zeros((4,2),dtype = np.float32)

  add = h.sum(1)
  hnew[0] = h[np.argmin(add)]
  hnew[2] = h[np.argmax(add)]
   
  diff = np.diff(h,axis = 1)
  hnew[1] = h[np.argmin(diff)]
  hnew[3] = h[np.argmax(diff)]

  return hnew

def preprocess(img):
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray,(5,5),2 )
  thresh = cv2.adaptiveThreshold(blur,255,1,1,11,1)
  return thresh
  
def imgdiff(img1,img2):
  img1 = cv2.GaussianBlur(img1,(5,5),5)
  img2 = cv2.GaussianBlur(img2,(5,5),5)    
  diff = cv2.absdiff(img1,img2)  
  diff = cv2.GaussianBlur(diff,(5,5),5)    
  flag, diff = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY) 
  return np.sum(diff)  

def find_closest_card(training,img):
  features = preprocess(img)
  return sorted(training.values(), key=lambda x:imgdiff(x[1],features))[0][0]
  
def getCards(im, numcards=4):
  gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray,(1,1),1000)
  flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY) 
       
  contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

  contours = sorted(contours, key=cv2.contourArea,reverse=True)[:numcards]  

  for card in contours:
    peri = cv2.arcLength(card,True)
    approx = rectify(cv2.approxPolyDP(card,0.2*peri,True))   
    
    h = np.array([ [0,0],[449,0],[449,449],[0,449] ],np.float32)

    transform = cv2.getPerspectiveTransform(approx,h)
    warp = cv2.warpPerspective(im,transform,(450,450))
    
    yield warp


def get_training(training_labels_filename,training_image_filename,num_training_cards,avoid_cards=None):
  training = {}
  
  labels = {}
  for line in file(training_labels_filename): 
    key, num, suit = line.strip().split()
    labels[int(key)] = (num,suit)
    
  print "Training"

  im = cv2.imread(training_image_filename)
  for i,c in enumerate(getCards(im,num_training_cards)):
    if avoid_cards is None or (labels[i][0] not in avoid_cards[0] and labels[i][1] not in avoid_cards[1]):
      training[i] = (labels[i], preprocess(c))
  
  print "Done training"
  return training

def mapcolor(src, hue, tolerance):
    """Creates a map with the locations of +/-tolerance hue"""
    imgHSV = cv2.cvtColor(src, cv2.COLOR_BGR2HSV) #Convert the captured frame from BGR to HSV
    iLowH = hue-tolerance
    iHighH = hue+tolerance
    iLowS = 100
    iHighS = 255
    iLowV = 100
    iHighV = 200
    out = cv2.inRange(imgHSV, np.array([iLowH, iLowS, iLowV]), np.array([iHighH, iHighS, iHighV])) #Threshold the image
    kernel = np.ones((5,5),np.uint8)
   #morphological opening (removes small objects from the foreground)
    out = cv2.erode(out, kernel, iterations=1 )
    out = cv2.dilate( out, kernel, iterations=1 )
   #morphological closing (removes small holes from the foreground)
    out = cv2.dilate( out, kernel, iterations=1 )
    out = cv2.erode(out, kernel, iterations=1 )
    return out



def mapshade( src, value, tolerance):
    """Creates a map wit the locations of +/-tolerance value"""
    imgHSV = cv2.cvtColor(src, cv2.COLOR_BGR2HSV) #Convert the captured frame from BGR to HSV
    iLowH = 1
    iHighH = 179
    iLowS = 1
    iHighS = 255
    iLowV = value-tolerance
    iHighV = value+tolerance
    out = cv2.inRange(imgHSV, np.array([iLowH, iLowS, iLowV]), np.array([iHighH, iHighS, iHighV])) #Threshold the image
    kernel = np.ones((1,1),np.uint8)
  #morphological opening (removes small objects from the foreground)
    out = cv2.erode(out, kernel, iterations=1 )
    out = cv2.dilate( out, kernel, iterations=1 )
   #morphological closing (removes small holes from the foreground)
    out = cv2.dilate( out, kernel, iterations=1 )
    out = cv2.erode(out, kernel, iterations=1 )
    return out


def countshapes(src):
    """Returns the number of shapes in the map src"""
    #finding all contours in the image.
    shapes = 0
    n = 0
    contour, hierarchy = cv2.findContours(src,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    while (n < len(contour)):
      tmoment = cv2.moments(contour[n])
      shape = cv2.approxPolyDP(contour[n], cv2.arcLength(contour[n], 0)*0.1, 0)
      area = tmoment['m00']
      if (area > 20):
        shapes = shapes + 1
      n=n+1
    return len(contour)


def findshapex(src, n):
    """Returns the central point of the nth shape in map src"""
    temp1 = src
    #finding all contours in the image.
    contour, hierarchy = cv2.findContours(src,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    tmoment = cv2.moments(contour[n])
    shape = cv2.approxPolyDP(contour[n], cv2.arcLength(contour[n], 0)*0.1, 0)
    x = tmoment['m10']/tmoment['m00']
    y = tmoment['m01']/tmoment['m00']
    return x

def findshapey(src, n):
    """Returns the central point of the nth shape in map src"""
    temp1 = src
    #finding all contours in the image.
    contour, hierarchy = cv2.findContours(src,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    tmoment = cv2.moments(contour[n])
    shape = cv2.approxPolyDP(contour[n], cv2.arcLength(contour[n], 0)*0.1, 0)
    x = tmoment['m10']/tmoment['m00']
    y = tmoment['m01']/tmoment['m00']
    return y

def findarea(src, n):
    """Returns the area of the nth shape in map src"""
    temp1 = src
    #finding all contours in the image.
    contour, hierarchy = cv2.findContours(src,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    tmoment = cv2.moments(contour[n])
    shape = cv2.approxPolyDP(contour[n], cv2.arcLength(contour[n], 0)*0.1, 0)
    area = tmoment['m00']
    return area


def removecolor(src, hue, tolerance):
    """Removes the color given in hue"""
    imgThresholded = mapcolor(src, hue, tolerance)
    imgEdited = cv2.cvtColor(imgThresholded, cv2.COLOR_GRAY2BGR)
    imgGray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    imgGray = cv2.cvtColor(imgGray, cv2.COLOR_GRAY2BGR)
    imgGray = cv2.bitwise_and(imgGray, imgEdited)
    imgEdited = cv2.bitwise_and(src, imgEdited)
    imgEdited = src - imgEdited + imgGray
    return imgEdited

def waitforcard():
    """Moves the player of the given hue to another location"""
    ret, first = cap.read()
    first = mapshade(first, 20, 20)
    ret, second = cap.read()
    second = mapshade(second, 20, 20)
    while (first == second):
      ret, first = cap.read()
      first = mapshade(first, 20, 20)
      ret, second = cap.read()
      second = mapshade(second, 20, 20)
    while (first != second):
      ret, first = cap.read()
      first = mapshade(first, 20, 20)
      ret, second = cap.read()
      second = mapshade(second, 20, 20)
    return
  

def turn(data, player):
    """Runs a turn for player number 'player' """
    failure=1;
    while(failure):
        if player == 1:
            print "It is yellow\'s turn."
        elif player == 2:
            print "It is green\'s turn."
        elif player == 3:
            print "It is teal\'s turn."
        elif player == 4:
            print "It is blue\'s turn."
        elif player == 5:
            print "It is purple\'s turn."
        elif player == 6:
            print " It is red\'s turn."
       # ret, startturn = cap.read()
       # cv2.imshow("New Turn", startturn)
       # cv2.waitKey(0)
       # ret, endturn = cap.read()
       # numberofcards = 18
       # training = get_training('cards.tsv','cards.jpg',numberofcards)
       # cards = [find_closest_card(training,c) for c in getCards(endturn,1)]
       ## waitforcard()
       ## ret, startturn = cap.read()
       ## startturn = mapshade(startturn, 20, 20)
       ## cards = countshapes(startturn)
        cards = int(raw_input('What is your action?'))
        if cards == 1: #Move
            data[player][0] = data[player][0] + movement(player*30, data[player][3], data[player][4])
            failure=0
        elif cards == 2: #Fire
            if data[player][2]>=1:
                targethue=huetarget(player*30)
                if (fire(player*30, targethue)==1):
                    targetplayer = targethue/30
                    if targethue==0:
                        targetplayer=6
                    data[targetplayer][1]=data[targetplayer][1]-1
                data[player][0]=data[player][0]+1
                data[player][2]=data[player][2]-1
            failure=0
        elif cards == 3: #Reload
            data[player][2]=6
            data[player][0]=data[player][0]+3
            failure=0
        else:
            print "Error, misread action card. Please try again.\n"


def movement(hue, speedx, speedy):
    """Moves the player of the given hue to another location"""
    ret, startturn = cap.read()
    print "You have chosen to move.\nPlease move as far as you want and press Enter.\n";
    temp = raw_input()
    ret, endturn = cap.read()
    endturn = mapcolor(endturn, hue, 10)
    startturn = mapcolor(startturn, hue, 10)
    xi = findshapex(startturn, 0)
    yi = findshapey(startturn, 0)
    xf = findshapex(endturn, 0)
    yf = findshapey(endturn, 0)
    area = findarea(startturn, 0)
    distance = math.sqrt(((xi-xf)*(xi-xf))+((yi-yf)*(yi-yf)))
    speedx = xi-xf
    speedy = yi-yf
    print "You moved a distance of ", math.ceil(distance/math.sqrt(area)), ".\n"
    return math.ceil(10*distance/math.sqrt(area))


def huetarget(hue):
    """Allows the player to target a particular hue based on whichever is closest to the card"""
    while(Failure):
        cap.read(startturn)
        test = mapshade(startturn, 21, 20)
        targetlocation = findshape(startturn, 0)
        test = mapcolor(startturn, hue, 10)
        shooterlocation = findshape(test, 0)
        while n<=5:
            test = mapcolor(startturn, (hue+n*30)%180, 10)
            testlocation = findshape(test, 0)
            testdistance = sqrt(((targetlocation.x-testlocation.x)*(targetlocation.x-testlocation.x))+((targetlocation.y-testlocation.y)*(targetlocation.y-testlocation.y)))
            if testdistance<distance:
                distance = testdistance
                targethue = (hue+n*30)%180
            if targethue == 30:
                print "You fired at yellow"
            elif targethue == 60:
                print "You fired at green"
            elif targethue == 90:
                print "You fired at teal"
            elif targethue == 120:
                print "You fired at blue"
            elif targethue == 150:
                print "You fired at purple"
            elif targethue == 0:
                print "You fired at red"
            n=n+1
        Failure=0
    return targethue


def testforplayer(n):
    """Test to see if there is a token for player n on the board"""
    _, src = cap.read()
    out = mapcolor(src, (n*30)%180, 10)
    cv2.imshow("test", src)
    cv2.imshow("map", out)
    k = cv2.waitKey() & 0xFF
    shapes = countshapes(out)
    return shapes



#Main sequence
#data = timer (0), health (1), ammo (2), speedx (3), speedy (4)

data = [[1],[1, 100, 6, 0, 0],[1, 100, 6, 0, 0],[1, 100, 6, 0, 0],[1, 100, 6, 0, 0],[1, 100, 6, 0, 0],[1, 100, 6, 0, 0]]

cap = cv2.VideoCapture(1)
engine = pyttsx.init()
engine.say('Welcome to the game.')
engine.runAndWait()
while data[0][0]<=100:
    n=1
    print "\nIT IS TURN ", data[0][0], "!\n\n"
    while n<=5:
        if ((data[0][0]==data[n][0]) & (testforplayer(n)>0) & (data[n][1]>0)):
            turn(data, n)
        else:
            print "No "
            if n == 1:
                print "yellow"
            elif n == 2:
                print "green"
            elif n == 3:
                print "teal"
            elif n == 4:
                print "blue"
            elif n == 5:
                print "purple"
            elif n == 6:
                print "red"
            print " player."
            cv2.destroyAllWindows()
        n=n+1
    data[0][0]=data[0][0]+1
    k = cv2.waitKey() & 0xFF
    if k == 27:
        break

print "GAME OVER!"
cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()
