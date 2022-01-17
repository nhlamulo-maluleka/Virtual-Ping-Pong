import cv2 as cv
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import sounddevice as sd
import soundfile as sf
from threading import Thread
import random
import time

# Extract data and sampling rate from file
array, smp_rt = sf.read('hit.wav', dtype = np.int16)

window = "Ping Pong"
cap = cv.VideoCapture(0)
cap.set(3, 1220)
cap.set(4, 720)

detector = HandDetector(maxHands=1)
colors = [(88, 75, 5), (64, 56, 228), (139, 2, 40), (17, 142, 107), (219, 40, 239)]

# Delta Variables
ballMovement = 15
dy, dx = ballMovement, ballMovement

# Movement Variables
aixpos, aiypos, increase, aispeed = 150, 290, 2, 10
xpos, ypos, pls, ple, plmove = 610, 360, 60, 200, 10
sdelay, score, ai_score, spoint, points, next = 0, 0, 0, 0, [], []
placeAICentre = 15
auto = True

# Color selection
choice = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# Model Variables
m, c = None, None

# Live stream toggle
liveView, p_next = False, False

# Start time
startT, prevTime = time.time(), -1

def hit(arr, smRate):
    sd.play(arr, smRate)

def AiPlayer(frame, start, end):
    cv.rectangle(frame, (frame.shape[1] - 25, start), (frame.shape[1] - 10, end), (0, 165, 255), cv.FILLED)

def manualPlayer(frame, start, end):
    cv.rectangle(frame, (5, start), (20, end), (150, 120, 178), cv.FILLED)

def movePlayer(frame, pls, ple, plmove, value):
    center = (0, value)

    if center[1] < pls:
        pls, ple = pls - plmove, ple - plmove
    elif center[1] > ple:
        pls, ple = pls + plmove, ple + plmove
    
    mid = ple - 70
    if mid < center[1]-placeAICentre or center[1]+placeAICentre < mid:
        if center[1]+placeAICentre < mid:
            pls, ple = pls - plmove, ple - plmove
        elif mid < center[1]-placeAICentre:
            pls, ple = pls + plmove, ple + plmove

    if pls < 60:
        pls, ple = 60, 200
    elif ple > frame.shape[0]:
        pls, ple = frame.shape[0] - 140, frame.shape[0]
    
    return pls, ple


def moveAi(frame, aixpos, aiypos, aispeed, value):
    center = (frame.shape[1], value)

    if center[1] < aixpos:
        aixpos, aiypos = aixpos - aispeed, aiypos - aispeed
    elif center[1] > aiypos:
        aixpos, aiypos = aixpos + aispeed, aiypos + aispeed
    
    mid = aiypos - 70
    if mid < center[1]-placeAICentre or center[1]+placeAICentre < mid:
        if center[1]+placeAICentre < mid:
            aixpos, aiypos = aixpos - aispeed, aiypos - aispeed
        elif mid < center[1]-placeAICentre:
            aixpos, aiypos = aixpos + aispeed, aiypos + aispeed

    if aixpos < 60:
        aixpos, aiypos = 60, 200
    elif aiypos > frame.shape[0]:
        aixpos, aiypos = frame.shape[0] - 140, frame.shape[0]

    return aixpos, aiypos

############ Prediction Model Section ##################

def getConstant(m, points):
    return int(points[1] - m*points[0])

def getMC(points):
    x1, y1 = points[1]
    x2, y2 = points[0]

    m = int(np.divide(y2 - y1, x2 - x1))
    c = getConstant(m, points[0])

    return m, c

def predictY(m, c, x):
    return int(m*x + c)

def predictX(m, c, y):
    return int(np.divide(y - c, m))

def predictNextPoints(frame, dx, dy, next, xpos, ypos, hEndPoint):
    next.append((xpos + dx*4, ypos + dy*4))
    next.append((xpos + dx*6, ypos + dy*6))
    
    # Gradient + Constant
    m, c = getMC(next) 

    cy_Val = predictY(m, c, hEndPoint)

    if cy_Val < 60 or cy_Val > frame.shape[0]:    
        temp_dy = dy

        if dy > 0:
            # Down
            t_ypos = frame.shape[0]
            x = predictX(m, c, t_ypos)
            temp_dy *= -1
            next.clear()

            next.append((x + dx*4, t_ypos + temp_dy*4))
            next.append((x + dx*6, t_ypos + temp_dy*6))

        else:
            # Up
            t_ypos = 60
            x = predictX(m, c, t_ypos)
            temp_dy *= -1
            next.clear()

            next.append((x + dx*4, t_ypos + temp_dy*4))
            next.append((x + dx*6, t_ypos + temp_dy*6))

        # cv.line(frame, next[0], (x, t_ypos), (255, 255, 255), 1)
        
    # cv.line(frame, next[0], (hEndPoint, cy_Val), (255, 255, 255), 1)

    return next, False

################################################################

while True:    
    _, img = cap.read()
    frame = img.copy()
    hands, frame = detector.findHands(frame)
    if not liveView: frame = np.zeros_like(frame)
    x_value = None

    cv.rectangle(frame, (0, 0), (frame.shape[1], 60), (255, 255, 255), cv.FILLED)
    cv.putText(frame, f"AI Player: {ai_score}", (frame.shape[1] - 300, 40), cv.FONT_HERSHEY_SIMPLEX, .9, (0, 0, 0), 2)
    cv.putText(frame, f"Player: {score}", (180, 40), cv.FONT_HERSHEY_SIMPLEX, .9, (0, 0, 0), 2)
    cv.putText(frame, f"Speed: x{ballMovement}", (int(frame.shape[1]//2)-50, 40), cv.FONT_HERSHEY_SIMPLEX, .9, (0, 0, 0), 2)

    if hands and not auto:
        fingers = detector.fingersUp(hands[0])

        # Move Up
        if fingers == [1, 1, 0, 0, 0]:
            pls, ple = pls - plmove, ple - plmove
        # Move Down
        elif fingers == [1, 0, 0, 0, 0]:
            pls, ple = pls + plmove, ple + plmove

        if pls <= 60:
            pls, ple = 60, 200
        elif ple >= frame.shape[0]:
            pls, ple = frame.shape[0] - 140, frame.shape[0]

    # Ball Movement
    xpos, ypos = xpos + dx, ypos + dy

    # Re-aligning the vertical position of the ball
    if ypos > frame.shape[0]:
        ypos = frame.shape[0]

    # AI Player Hit/Miss detection
    if xpos >= frame.shape[1] - 30:
        if aixpos <= ypos <= aiypos and sdelay == 0:
            Thread(target=hit, args=(array, smp_rt), daemon=True).start()
            xpos = frame.shape[1] - 30
            points.clear()
            dx *= -1            
            choice = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        elif sdelay == 0:
            score += 5
        sdelay = 1

    # Player Hit or Miss detection
    if xpos <= 25:
        if pls <= ypos <= ple and sdelay == 0:
            Thread(target=hit, args=(array, smp_rt), daemon=True).start()
            xpos = 25
            points.clear()
            dx *= -1
            choice = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        elif sdelay == 0:
            ai_score += 5
        sdelay = 1
    
        # Ball Display
    cv.circle(frame, (xpos, ypos), 15, choice, cv.FILLED)
    cv.circle(frame, (xpos, ypos), 15, (215, 255, 255), 1)

    # Delay for 10 seconds before updating the score again
    if sdelay >= 1:
        sdelay += 1
        if sdelay > 10: sdelay = 0
    
    if xpos > 30:
        spoint += 1

    # If points then find the slope
    if len(points) == 2:
        if not m and not c:
            m, c = getMC(points)

        x_point = frame.shape[1] if dx > 0 else 0
        value = predictY(m, c, x_point)

        if value < 60:
            value = 60
            x_value = predictX(m, c, value)
        elif value > frame.shape[0]:
            value = frame.shape[0]
            x_value = predictX(m, c, value)
        
        # cv.circle(frame, (x_point, value), 5, (234, 123, 234), cv.FILLED)

        if dx > 0: 
            aixpos, aiypos = moveAi(frame, aixpos, aiypos, aispeed, value)
        elif dx < 0 and auto: 
            pls, ple = movePlayer(frame, pls, ple, plmove, value)

    # Player
    manualPlayer(frame, pls, ple)

    # AI Based player
    AiPlayer(frame, aixpos, aiypos)

    if xpos <= 0 or xpos >= frame.shape[1]:        
        dx *= -1
        points.clear()
        m, c = None, None
        if xpos > 25: 
            p_next = True

    if ypos <= 80 or ypos >= frame.shape[0]:
        dy *= -1
        points.clear()
        m, c = None, None
        p_next = True

    # Predict next to movements
    if p_next:
        points, p_next = predictNextPoints(frame, dx, dy, next, xpos, ypos, frame.shape[1] if dx > 0 else 0)

    t = int(np.mod((time.time() - startT), 60))
    cv.putText(frame, f"Time: {t}", (10, 35), cv.FONT_HERSHEY_COMPLEX, .7, 2)

    if increase == 0:
        ballMovement += 1

        if dx > 0: dx += 1
        else: dx -= 1

        if dy > 0: dy += 1 
        else: dy -= 1

        increase = 2
        prevTime = -1
    
    if t != prevTime:
        prevTime = t
        if t == 0: increase -= 1

    cv.namedWindow(window, cv.WINDOW_NORMAL)
    cv.setWindowProperty(window, cv.WINDOW_NORMAL, cv.WINDOW_FULLSCREEN)
    cv.imshow(window, frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        cv.destroyAllWindows()
        break
    elif key == ord('t'):
        if liveView: liveView = False
        elif not liveView: liveView = True
    elif key == ord('i'):
        if dx > 0: dx += 1
        else: dx -= 1

        if dy > 0: dy += 1 
        else: dy -= 1

        ballMovement = dx

        if ballMovement < 0: ballMovement *= -1

    elif key == ord('d'):
        if dx > 0: dx -= 1
        else: dx += 1

        if dy > 0: dy -= 1 
        else: dy += 1

        ballMovement = dx
        if ballMovement < 0: ballMovement *= -1

    elif key == ord('m'):
        auto = False
        plmove = 15

    elif key == ord('a'):
        auto = True
        plmove = aispeed
