import cv2
import mediapipe as mp
import ctypes
import numpy as np


class HeliosPoint(ctypes.Structure):
    _fields_ = [
        ('x', ctypes.c_uint16),
        ('y', ctypes.c_uint16),
        ('r', ctypes.c_uint8),
        ('g', ctypes.c_uint8),
        ('b', ctypes.c_uint8),
        ('i', ctypes.c_uint8)
    ]

HeliosLib = ctypes.cdll.LoadLibrary(".\\HeliosLaserDAC.dll")
numDevices = HeliosLib.OpenDevices()

testColor = (50, 0, 0, 50)
blankColor = (0, 0, 0, 0)
confidence = 0.5
numBlankTansitionPoints = 30
numVisibleTransitionPoints = 30

mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture(0)


def MapToLaserRange(value, maxValue, invert=False):
    laserValue = int((value / maxValue) * 4095)
    if invert:
        return 4095 - laserValue
    return laserValue


def BlankTransitionCreate(start, end, points=numBlankTansitionPoints):
    interpolatedPoints = []

    fractions = np.linspace(0, 1, points)

    for t in fractions:
        interpolatedX = int(start[0] + t * (end[0] - start[0]))
        interpolatedY = int(start[1] + t * (end[1] - start[1]))

        newPoint = HeliosPoint(
            x=interpolatedX,
            y=interpolatedY, 
            r=blankColor[0],
            g=blankColor[1],
            b=blankColor[2],
            i=blankColor[3]
        )
        interpolatedPoints.append(newPoint)

    return interpolatedPoints

def VisiblePointsCreate(start, end, points=numVisibleTransitionPoints):
    interpolatedPoints = []

    fractions = np.linspace(0, 1, points)

    for t in fractions:
        interpolatedX = int(start[0] + t * (end[0] - start[0]))
        interpolatedY = int(start[1] + t * (end[1] - start[1]))

        newPoint = HeliosPoint(
            x=interpolatedX,
            y=interpolatedY, 
            r=testColor[0],
            g=testColor[1],
            b=testColor[2],
            i=testColor[3]
        )
        
        interpolatedPoints.append(newPoint)

    return interpolatedPoints

def HeadCreate(center, radius, numHeadPoints=20):

    headPoints = []
    startAngle = 3 * np.pi / 2
    stepAngle = 2 * np.pi / numHeadPoints
    
    for i in range(numHeadPoints + 1):  # +1 to close circle
        angle = startAngle + i * stepAngle
        circleX = int(center[0] + radius * np.cos(angle))
        circleY = int(center[1] + radius * np.sin(angle))
        
        if abs(circleY - center[1]) <= radius:
            newPoint = HeliosPoint(
                x=circleX,
                y=circleY, 
                r=testColor[0],
                g=testColor[1],
                b=testColor[2],
                i=testColor[3]
            )
            
            headPoints.append(newPoint)
        
    return headPoints
 

def StickmanCreate(landmarks, w, h):
    keypoints = []
    if len(landmarks) == 0:
        return keypoints

    noseID = 0
    lShoulderID = 11
    rShoulderID = 12
    lElbowID = 13
    rElbowID = 14
    lWristID = 15
    rWristID = 16
    lHipID = 23
    rHipID = 24
    lKneeID = 25
    rKneeID = 26
    lAnkleID = 27
    rAnkleID = 28

    def LaserMap(lm):
        return (
            MapToLaserRange(lm.x * w, w),
            MapToLaserRange(lm.y * h, h, True)
        )

    nose = landmarks[noseID]
    lShoulder = landmarks[lShoulderID]
    rShoulder = landmarks[rShoulderID]
    lHip = landmarks[lHipID]
    rHip = landmarks[rHipID]

    headPoints = []
    torsoLine = None
    neckPoint = None
    hipPoint = None

    if nose.visibility > confidence:
        noseMapped = LaserMap(nose)
        headRadius = 300
        headPoints = HeadCreate(noseMapped, headRadius, 20)
    else:
        return keypoints

    if lShoulder.visibility > confidence and rShoulder.visibility > confidence:
        x = (lShoulder.x + rShoulder.x) / 2
        y = (lShoulder.y + rShoulder.y) / 2
        neckPoint = (
            MapToLaserRange(x * w, w),
            MapToLaserRange(y * h, h, True)
        )

    if lHip.visibility > confidence and rHip.visibility > confidence:
        x = (lHip.x + rHip.x) / 2
        y = (lHip.y + rHip.y) / 2
        hipPoint = (
            MapToLaserRange(x * w, w),
            MapToLaserRange(y * h, h, True)
        )

    if neckPoint and hipPoint:
        torsoLine = (neckPoint, hipPoint)

    stickmanLines = []

    if neckPoint:
        stickmanLines.extend([
            (neckPoint, lElbowID, lWristID),
            (neckPoint, rElbowID, rWristID)
        ])

    if hipPoint:
        stickmanLines.extend([
            (hipPoint, lKneeID, lAnkleID),
            (hipPoint, rKneeID, rAnkleID)
        ])

    visibleLines = []
    for startID, midID, endID in stickmanLines:
        mid = landmarks[midID]
        end = landmarks[endID]
        if mid.visibility > confidence:
            miidPoint = LaserMap(mid)
            visibleLines.append((startID, miidPoint))
            if end.visibility > confidence:
                endPoint = LaserMap(end)
                visibleLines.append((miidPoint, endPoint))


    keypoints += headPoints

    if torsoLine:
        startPoint, endPoint = torsoLine
        keypoints += VisiblePointsCreate(startPoint, endPoint)

    previousEndPoint = None
    for (startPoint, endPoint) in visibleLines:
        if previousEndPoint and previousEndPoint != startPoint:
            keypoints += BlankTransitionCreate(previousEndPoint, startPoint)
        keypoints += VisiblePointsCreate(startPoint, endPoint)
        previousEndPoint = endPoint

    if previousEndPoint and headPoints:
        keypoints += BlankTransitionCreate(previousEndPoint, (headPoints[0].x, headPoints[0].y))

    return keypoints


def HeliosFrameFill(frame, points, nosePoint=20):
    numVisiblePoints = min(len(points), 1000)
    for i in range(numVisiblePoints):
        frame[i] = points[i]

    if numVisiblePoints < 1000:
        if points:
            lastPoint = (points[-1].x, points[-1].y)
        else:
            lastPoint = nosePoint

        numBlankPoints = 1000 - numVisiblePoints

        blankPoints = BlankTransitionCreate(lastPoint, nosePoint, numBlankPoints)

        for i, blankPoint in enumerate(blankPoints, start=numVisiblePoints):
            frame[i] = blankPoint


galvoSpeed = 30 * 1000  # points per second

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = pose.process(rgbFrame)

    hFrame, wFrame = frame.shape[0], frame.shape[1]
    frameType = HeliosPoint * 1000
    heliosFrame = frameType()

    keypoints = []
    if result.pose_landmarks:
        keypoints = StickmanCreate(result.pose_landmarks.landmark, wFrame, hFrame)

        if keypoints:
            nextFrameStart = (keypoints[0].x, keypoints[0].y) if keypoints else (2048,2048)

        HeliosFrameFill(heliosFrame, keypoints, nextFrameStart)
    else:
        HeliosFrameFill(heliosFrame, [], (2048,2048))

    for point in keypoints:
        if point.i != 0:
            normalizedX = point.x / 4095.0
            normalizedY = point.y / 4095.0

            x = int(normalizedX * wFrame)
            y = int((1.0 - normalizedY) * hFrame)

            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)


    for i in range(numDevices):
        statusAttempts = 0
        while statusAttempts < 512 and HeliosLib.GetStatus(i) != 1:
            statusAttempts += 1
        HeliosLib.WriteFrame(i, galvoSpeed, 0, ctypes.pointer(heliosFrame), 1000)

    cv2.imshow('Mediapipe preview', frame)

    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
HeliosLib.CloseDevices()
