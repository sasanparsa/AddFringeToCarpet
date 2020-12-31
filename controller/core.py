import os
import traceback
from flask import Flask, request, render_template
import json
from copy import deepcopy
import cv2
from PIL import Image
import base64
from io import BytesIO
import numpy as np
import math
import glob
import random
import pyodbc

class Logger:
    def __init__(self):
        self.Logs = []
    def Log(self,text):
        self.Logs.append(text)
        print(text)
    def GetLogsText(self):
        res= ''
        for i in self.Logs:
            res += i
            res += '    \n'
        return res

# core.py directory path config
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path.replace("\\", "/")

# set pages path into template_folder and files path into static_folder
app = Flask(__name__, template_folder=dir_path + '/..', static_folder=dir_path + '/../files')


# -----------------------GET---------------------------------------------------------------
@app.route('/')
def index():
    return render_template('views/DrawFringeOnCarpet.html')


@app.route('/addfringe')
def addfringe():
    return render_template('views/AddNewFringe.html')

logger = Logger()
# -----------------------POST---------------------------------------------------------------
@app.route('/Fringe', methods=['POST'])
def Fringe():
    try:
        picture = request.files['picture']
        picture_b64 = request.form['picture_b64']
        inputPicBase64 = picture_b64
        fringe = request.form['fringe']
        fringe_size_cm = fringe.split("_")[1]
        fringe_size_cm = fringe_size_cm[0:fringe_size_cm.index('.png')].split('S')
        fringe_width_cm,fringe_height_cm = float(fringe_size_cm[0]),float(fringe_size_cm[1])

        logger.Log('fringe_width_cm: '+ str(fringe_width_cm))
        logger.Log('fringe_height_cm: ' + str(fringe_height_cm))

        # create image from fringe path
        fringe = cv2.imread(fringe, cv2.IMREAD_UNCHANGED)

        # create image from input stream
        picture = cv2.imdecode(np.fromstring(picture.read(), np.uint8), cv2.IMREAD_UNCHANGED)

        # ignore alpha channel of png files
        if (len(picture[0][0]) == 4):
            picture = picture[:, :, 0:3]

        # find Aruco Corners
        ArucoCorners = DetectArucoCorners(picture)
        if (len(ArucoCorners) == 0):
            return json.dumps({'success': 'false', 'message': 'can not find tag on carpet'})
        (aruco_tl, aruco_tr, aruco_br, aruco_bl) = OrderPointsClockWise(ArucoCorners)
        # DrawPointsOnPicture(picture,ArucoCorners)

        CarpetCorners = np.array(DetectCarpetCorners(picture, ArucoCorners))
        logger.Log('Carpet Corners: '+str(CarpetCorners))
        (carpet_tl,carpet_tr,carpet_br,carpet_bl) = OrderPointsClockWise(CarpetCorners)

        # carpet_bottom_edge_px = TwoPointsDistance(carpet_br,carpet_bl)
        # carpet_top_edge_px = TwoPointsDistance(carpet_tr, carpet_tl)
        # carpet_right_edge_px = TwoPointsDistance(carpet_br, carpet_tr)
        # carpet_left_edge_px = TwoPointsDistance(carpet_tl, carpet_bl)

        DrawPointsOnPicture(picture, CarpetCorners)
        twod_picture = FourPointTransform(picture, np.asarray(CarpetCorners))
        ShowImage(twod_picture)


        twod_picture  = cv2.resize(twod_picture, (round(max(TwoPointsDistance(carpet_tl,carpet_tr),
                                                            TwoPointsDistance(carpet_br,carpet_bl)) * 2)
                                                  , round(max(TwoPointsDistance(carpet_tl,carpet_bl),
                                                            TwoPointsDistance(carpet_tr,carpet_br))*2)),
                                                        interpolation=cv2.INTER_CUBIC)
        ShowImage(twod_picture)

        ArucoCorners2 = DetectArucoCorners(twod_picture)
        if (len(ArucoCorners2) == 0):
            return json.dumps({'success': 'false', 'message': 'can not find tag on carpet'})
        aruco_cm = 25.0
        (aruco2_tl,aruco2_tr,aruco2_br,aruco2_bl) = OrderPointsClockWise(ArucoCorners2)

        twod_carpet_width_cm = (len(twod_picture[0]) / TwoPointsDistance(aruco2_tl,aruco2_tr))*(aruco_cm)
        twod_carpet_height_cm = (len(twod_picture) / TwoPointsDistance(aruco2_bl, aruco2_tl)) * (aruco_cm)
        original_carpet_width_cm = carpet_cm = min(twod_carpet_height_cm,twod_carpet_width_cm)
        original_carpet_height_cm = max(twod_carpet_height_cm, twod_carpet_width_cm)

        logger.Log('carpet_height: '+str(original_carpet_height_cm))
        logger.Log('carpet_width: ' + str(original_carpet_width_cm))

        twod_tl = [0,0]
        twod_tr = [len(twod_picture[0]),0]
        twod_br = [len(twod_picture[0]),len(twod_picture)]
        twod_bl = [0,len(twod_picture)]

        distance_aruco_from_left = TwoPointsDistance(aruco2_tl, twod_tl) + TwoPointsDistance(aruco2_bl, twod_bl)
        distance_aruco_from_right = TwoPointsDistance(aruco2_tr, twod_tr) + TwoPointsDistance(aruco2_br, twod_br)
        distance_aruco_from_bottom = TwoPointsDistance(aruco2_br, twod_br) + TwoPointsDistance(aruco2_bl, twod_bl)
        distance_aruco_from_top = TwoPointsDistance(aruco2_tr, twod_tr) + TwoPointsDistance(aruco2_tl, twod_tl)

        list_of_distances = [distance_aruco_from_top,distance_aruco_from_bottom,distance_aruco_from_right,distance_aruco_from_left]

        logger.Log('list_of_distances: '+str(list_of_distances))
        idx = list_of_distances.index(min(list_of_distances))
        aruco_max_width_px = max(TwoPointsDistance(aruco_tr,aruco_tl),TwoPointsDistance(aruco_br,aruco_bl))
        aruco_max_height_px = max(TwoPointsDistance(aruco_br, aruco_tr), TwoPointsDistance(aruco_bl, aruco_tl))

        DrawPointsOnPicture(picture,[aruco_tl,aruco_tr,aruco_br,aruco_bl])

        if (idx == 0 or idx == 1):
            fringe_width_pixel_per_cm = aruco_max_width_px / aruco_cm
            fringe_height_pixel_per_cm = aruco_max_height_px / aruco_cm
            if(idx == 0):
                (p1, p2, p3, p4) = (carpet_br, carpet_tr, carpet_tl, carpet_bl)     #top

            else:
                (p1, p2, p3, p4) = (carpet_tr, carpet_br, carpet_bl, carpet_tl)     #bottom
                fringe_height_pixel_per_cm += (fringe_height_pixel_per_cm*0.65)
                fringe_width_pixel_per_cm += (fringe_width_pixel_per_cm * 0.65)
        else:
            fringe_width_pixel_per_cm = aruco_max_height_px / aruco_cm
            fringe_height_pixel_per_cm = aruco_max_width_px / aruco_cm
            aruco_distance_from_left_and_right = (original_carpet_width_cm - aruco_cm) /2
            fringe_width_pixel_per_cm += (aruco_distance_from_left_and_right * 0.017)
            fringe_height_pixel_per_cm += (aruco_distance_from_left_and_right * 0.017)
            if(idx == 2):
                (p1, p2, p3, p4) = (carpet_tl, carpet_tr, carpet_br, carpet_bl)     #right
            else:
                (p1, p2, p3, p4) = (carpet_tr, carpet_tl, carpet_bl, carpet_br)     #left

        fringew = int(round(fringe_width_pixel_per_cm * fringe_width_cm))
        fringeh = int(round(fringe_height_pixel_per_cm * fringe_height_cm))

        # DrawPointsOnPicture(twod_picture, [aruco_tr, aruco_tl], True, (255, 0, 0))
        DrawPointsOnPicture(picture, [p2,p3], True, (255, 0, 0),25)

        (x1,y1,x2,y2,x3,y3,x4,y4) = (int(round(p1[0])),int(round(p1[1])),int(round(p2[0])),int(round(p2[1])),
                                     int(round(p3[0])), int(round(p3[1])),int(round(p4[0])),int(round(p4[1])))

        # server side validation
        if ((x1 < 0 or x1 > 1024) or (x2 < 0 or x2 > 1024) or (x3 < 0 or x3 > 1024) or (x4 < 0 or x4 > 1024) or (
                y1 < 0 or y1 > 5000) or
                (y2 < 0 or y2 > 5000) or (y3 < 0 or y3 > 5000) or (y4 < 0 or y4 > 5000) or (len(fringe) == 0)):
            return json.dumps({'success': 'false', 'message': 'please select 4 vertices'})
        if (carpet_cm <= 40 or carpet_cm > 500):
            return json.dumps(
                {'success': 'false', 'message': 'please enter a valid value for carpet size in cm(50 to 500 cm'})

        if ((len(picture) < 200 or len(picture) > 5000) or (len(picture[0]) < 200 or len(picture[0]) > 1024)):
            return json.dumps({'success': 'false', 'message': 'please take picture in normal distance from carpet'})
            # define variables
        min_carpet_length_for_perspective = 150

        # compute left slope and right slope and carpet`s line slope
        shib_right = CalculateSlope([x1,y1],[x2,y2])
        shib_left = CalculateSlope([x4,y4],[x3,y3])
        shib_khat = CalculateSlope([x3,y3],[x2,y2])

        logger.Log("before retation:")
        logger.Log("shib_khat:" + str(shib_khat))
        logger.Log("shib_left:" + str(shib_left))
        logger.Log("shib_right:" + str(shib_right))

        if (abs(shib_khat) >= 2):
            perspective_rate = 0.5
        else:
            perspective_rate = (abs(shib_khat) / 2) * 0.5

        # check if the right and left slope has equal sign of slope
        if (sign(shib_right) != sign(shib_left)):
            mode = "ver"
        else:
            mode = "hor"

        # compute rot_number for rotate carpet picture
        rot_number = 0
        if (shib_khat < 1 and shib_khat > -1):
            if (y1 == y2): y1 += sign(shib_khat) * sign(x2 - x1)
            if (y3 == y4): y4 += sign(shib_khat) * sign(x3 - x4)
            if ((y1 > y2) and (y4 > y3)):
                rot_number = -1
            elif ((y1 < y2) and (y4 < y3)):
                rot_number = 1
        else:
            if (x1 == x2): x1 += sign(shib_khat) * sign(y2 - y1)
            if (x3 == x4): x4 += sign(shib_khat) * sign(y3 - y4)
            if ((x1 > x2) and (x4 > x3)): rot_number = 2

        if (mode == "hor"):
            rot_number -= 1

        logger.Log("rot: "+str(rot_number))

        # if(abs(rot_number) == 2):
        #     return json.dumps({'success': 'false', 'message': 'please take picture in front of edge'})

        # rotate points coords
        (x1, y1, x2, y2, x3, y3, x4, y4) = rotate_points(x1, y1, x2, y2, x3, y3, x4, y4, rot_number, len(picture),
                                                         len(picture[0]))

        # rotate carpet picture
        picture = np.array(np.rot90(picture, rot_number))
        picture = deepcopy(picture)
        # compute left and right slopes again(after rotation)
        shib_right = CalculateSlope([x1,y1],[x2,y2])
        shib_left =  CalculateSlope([x4,y4],[x3,y3])

        if (mode == "hor"):
            khat_rish_vectors = abs(x2 - x3)
        else:
            khat_rish_vectors = abs(y2 - y3)

        if (khat_rish_vectors < 200):
            return json.dumps({'success': 'false', 'message': 'please come near to the carpet and take picture'})



        if (idx == 0):  # top
            # fringew = (carpet_top_edge_px * fringe_width_cm) / original_carpet_width_cm
            if (sign(shib_right) == 1):
                nearest_vertex_23 = 2
                # fringeh = (carpet_left_edge_px * fringe_height_cm) / original_carpet_height_cm
            else:
                nearest_vertex_23 = 3
                # fringeh = (carpet_right_edge_px * fringe_height_cm) / original_carpet_height_cm
        elif (idx == 1):  # bottom
            # fringew = (carpet_bottom_edge_px * fringe_width_cm) / original_carpet_width_cm
            if (sign(shib_right) == 1):
                nearest_vertex_23 = 3
                # fringeh = (carpet_left_edge_px * fringe_height_cm) / original_carpet_height_cm
            else:
                nearest_vertex_23 = 2
                # fringeh = (carpet_right_edge_px * fringe_height_cm) / original_carpet_height_cm
        elif (idx == 2):  # right
            nearest_vertex_23 = 3
            # fringew = (carpet_right_edge_px * fringe_width_cm) / original_carpet_width_cm
            # fringeh = (carpet_bottom_edge_px * fringe_height_cm) / original_carpet_height_cm
        elif (idx == 3):  # left
            nearest_vertex_23 = 2
            # fringew = (carpet_left_edge_px * fringe_width_cm) / original_carpet_width_cm
            # fringeh = (carpet_bottom_edge_px * fringe_height_cm) / original_carpet_height_cm

        # decrease points 2 and 3 for dents
        if (fringeh < 50):
            foroo_pix = 0
        elif (fringeh < 60):
            foroo_pix = 2
        elif (fringeh < 70):
            foroo_pix = 4
        elif (fringeh < 80):
            foroo_pix = 6
        elif (fringeh < 90):
            foroo_pix = 8
        else:
            foroo_pix = 10

        foroo_pix += 5

        logger.Log('foroo_pix: ' + str(foroo_pix))
        if (mode == "ver"):
            if (shib_right <= 1 and shib_right >= -1):
                x2 = x2 - foroo_pix
                y2 = get_y2_from_shib(shib_right, x1, y1, x2)
            else:
                y2 = y2 - (foroo_pix * sign(shib_right))
                x2 = get_x2_from_shib(shib_right, x1, y1, y2)

            if (shib_left <= 1 and shib_left >= -1):
                x3 = x3 - foroo_pix
                y3 = get_y2_from_shib(shib_left, x4, y4, x3)
            else:
                y3 = y3 - (foroo_pix * sign(shib_left))
                x3 = get_x2_from_shib(shib_left, x4, y4, y3)
        else:
            if (shib_right <= 1 and shib_right >= -1):
                x2 = x2 - (foroo_pix * sign(shib_right))
                y2 = get_y2_from_shib(shib_right, x1, y1, x2)
            else:
                y2 = y2 - foroo_pix
                x2 = get_x2_from_shib(shib_right, x1, y1, y2)

            if (shib_left <= 1 and shib_left >= -1):
                x3 = x3 - (foroo_pix * sign(shib_left))
                y3 = get_y2_from_shib(shib_left, x4, y4, x3)
            else:
                y3 = y3 - foroo_pix
                x3 = get_x2_from_shib(shib_left, x4, y4, y3)

        (x1, y1, x2, y2, x3, y3, x4, y4) = (
        max(x1, 0), max(y1, 0), max(x2, 0), max(y2, 0), max(x3, 0), max(y3, 0), max(x4, 0), max(y4, 0))
        shib_khat = CalculateSlope([x3,y3],[x2,y2])
        shib_khat_amood = -1 / (shib_khat + 0.001)

        logger.Log("after rotation:")
        logger.Log("shib_khat:" + str(shib_khat))
        logger.Log("shib_khat_amood:" + str(shib_khat_amood))
        logger.Log("shib_left:" + str(shib_left))
        logger.Log("shib_right:" + str(shib_right))

        fringeh += foroo_pix

        logger.Log('fringeh: '+str(fringeh))
        logger.Log('fringew: ' + str(fringew))
        fringe = cv2.resize(fringe, (round(fringew), round(fringeh)), interpolation=cv2.INTER_CUBIC)

        # cv2.line(picture, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.line(picture, (x3, y3), (x4, y4), (0, 255, 0), 2)
        # compute start point
        if (mode == "ver"):
            if (y3 < y2):
                startx = x3
                starty = y3
                endy = y2
                endx = x2
            else:
                startx = x2
                starty = y2
                endy = y3
                endx = x3
                shib_left, shib_right = shib_right, shib_left
        else:
            if (x3 < x2):
                startx = x3
                starty = y3
                endx = x2
                endy = y2
            else:
                startx = x2
                starty = y2
                endx = x3
                endy = y3
                shib_left, shib_right = shib_right, shib_left

        right_angle = round(math.atan(abs((shib_khat - shib_right) / (1 + shib_right * shib_khat))) * (180 / math.pi))
        left_angle = round(math.atan(abs((shib_khat - shib_left) / (1 + shib_left * shib_khat))) * (180 / math.pi))

        if (mode == "ver"):
            if (((sign(shib_left) == (sign(shib_khat) * -1)) and (shib_left > shib_khat_amood)) or
                    ((sign(shib_left) == (sign(shib_khat))) and (shib_left < shib_khat))):
                left_angle = 180 - left_angle
                right_angle = 180 - right_angle
        else:
            if (((sign(shib_left) == (sign(shib_khat) * -1)) and (shib_left < shib_khat_amood)) or
                    ((sign(shib_left) == (sign(shib_khat))) and (shib_left > shib_khat))):
                left_angle = 180 - left_angle
            if (((sign(shib_right) == (sign(shib_khat) * -1)) and (shib_right > shib_khat_amood)) or
                    ((sign(shib_right) == (sign(shib_khat))) and (shib_right < shib_khat))):
                right_angle = 180 - right_angle

        logger.Log("right_angle: "+str(right_angle))
        logger.Log("left_angle: " + str(left_angle))

        # if (mode == "ver" and (right_angle < 70 or right_angle > 110 or left_angle < 70 or left_angle > 110)):
        #     return json.dumps({'success': 'false', 'message': 'you probably are taking picture from near of carpet or ground. '
        #                             'please change your location and get away from carpet and ground or change your perspective'})
        # if (mode == "hor" and (right_angle < 50 or right_angle > 120 or left_angle < 50 or left_angle > 120)):
        #     return json.dumps({'success': 'false', 'message': 'you probably are taking picture from near of carpet or ground. '
        #                             'please change your location and get away from carpet and ground or change your perspective'})

        perspective_rate += min(abs(max(abs(right_angle - 90), abs(left_angle - 90))) / 40, 1) * 0.2
        if (perspective_rate > 0.5): perspective_rate = 0.5
        if(mode == 'ver' and (idx in [0,1])) : perspective_rate = 0.0


        logger.Log('perspective_rate: '+ str(perspective_rate))

        # drawing
        if (mode == "ver"):
            res = DrawFringeOnCarpet_Y(picture, fringe, startx, starty, endy, shib_left, shib_right, shib_khat,perspective_rate, nearest_vertex_23)
        else:
            res = DrawFringeOnCarpet_X(picture, fringe, startx, starty, endx, shib_left, shib_right, shib_khat,perspective_rate, nearest_vertex_23)
        # rotate drawed carpet picture vice versa
        res = np.rot90(res, rot_number * -1)
        logger.Log('----------------------------------')

        # create stream from carpet picture and return to the client
        result = Image.fromarray((cv2.cvtColor(res, cv2.COLOR_BGR2RGB)).astype(np.uint8), 'RGB')
        outputPicBase64 = ConverImageToBase64(result)
        print(outputPicBase64)
        db = DBManager()
        db.insert_request(inputPicBase64, outputPicBase64, logger.GetLogsText(), request.remote_addr)
        return json.dumps({'success': 'true', 'result': outputPicBase64})
    except Exception as e:
        traceback.print_exc()
        return json.dumps({'success': 'false', 'message': 'an error occured...change your location and try again...'})


@app.route('/GetAllFringes', methods=['POST'])
def GetAllFringes():
    # get fringe files paths and fringe picture stream and return to client
    list_of_file_names = glob.glob('../files/fringe/*.png')
    res = []
    for str in list_of_file_names:
        str = str.replace("\\", "/")
        res.append({"filename": str, "content": ConverImageToBase64(Image.open(str))})
    return json.dumps(res)


@app.route('/AddNewFringe', methods=['POST'])
def AddNewFringe():
    try:
        # read input from client
        fringe_cm = float(request.form['fringe_cm'])
        hasbg = request.form['hasbg']
        bgcolor = request.form['bgcolor']
        fringe = request.files['fringe']
        fringe = cv2.imdecode(np.fromstring(fringe.read(), np.uint8), cv2.IMREAD_UNCHANGED)

        # add alpha channel to rgb
        if (len(fringe[0][0]) == 3):
            b_channel, g_channel, r_channel = cv2.split(fringe)
            alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
            fringe = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

        # tresholding on fringe picture
        if (hasbg == 'yes'):
            for i in range(0, len(fringe)):
                for j in range(0, len(fringe[0])):
                    if ((bgcolor == "white" and (
                                fringe[i][j][0] > 128 and fringe[i][j][1] > 128 and fringe[i][j][2] > 128)) or
                            (bgcolor == "black" and (
                                        fringe[i][j][0] < 128 and fringe[i][j][1] < 128 and fringe[i][j][2] < 128))):
                        fringe[i][j][3] = 0

        filename = str(random.randint(1, 9999999999)) + '_' + str(fringe_cm) + '.png'
        cv2.imwrite("../files/fringe/" + filename, fringe)

        return json.dumps({"message": "new fringe added successfully"})
    except:
        traceback.print_exc()
        return json.dumps({"message": "some error occured.try again later..."})


@app.route('/StandardDimensions', methods=['POST'])
def StandardDimensions():
    try:
        # read input from client
        picture = request.files['picture']
        exif_rot = request.form['exif_rot']
        picture = cv2.imdecode(np.fromstring(picture.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        logger.Log('exif_rot: '+str(exif_rot))
        if (exif_rot == '6' or exif_rot == '7'):
            picture = np.rot90(np.array(picture), 3)
        elif (exif_rot == '3' or exif_rot == '4'):
            picture = np.rot90(np.array(picture), 2)
        elif (exif_rot == '5' or exif_rot == '8'):
            picture = np.rot90(np.array(picture), 1)

        pic_height = round(len(picture) * 1024 / len(picture[0]))
        picture = cv2.resize(picture, (1024, pic_height), interpolation=cv2.INTER_CUBIC)
        picture = Image.fromarray((cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)).astype(np.uint8), 'RGB')
        return json.dumps({"success": "true", "height": pic_height, "result": ConverImageToBase64(picture)})
    except:
        traceback.print_exc()
    return json.dumps({"message": "some error occured.try again later..."})


# --------------functions---------------------------------------------------------------------------------------------------------------

def DrawFringeOnCarpet_Y(img1, img2, x, y, y2, shib_left, shib_right, shib_khat, perspective_rate, nearest_vertex_23):
    xs, ys = x, y
    img1h, img1w, img2w = len(img1), len(img1[0]), len(img2[0])
    if (abs(shib_left) > 10): shib_left = 10 * sign(shib_left)
    if (abs(shib_right) > 10): shib_right = 10 * sign(shib_right)

    shib_rish = shib_left
    shib_diff = (shib_right - shib_left) / abs(y - y2)

    perspective_i = 0
    perspective_diff = perspective_rate * len(img2) / abs(y - y2)
    logger.Log('nearest_vertex_23: '+str(nearest_vertex_23))
    if (nearest_vertex_23 == 3):
        perspective_diff *= -1
        perspective_i = perspective_rate * len(img2)

    fringe_copy = deepcopy(img2)
    while (True):
        for i in range(ys, ys + img2w):
            if (i > y2 or i > img1h):
                return img1
            xx = get_x2_from_shib(shib_khat, xs, ys, i)
            xx2 = xx
            yy = i
            yy2 = yy

            ys_c = i
            xs_c = xx
            if (shib_rish <= 1 and shib_rish >= -1):
                for j in range(xs_c, xs_c + len(fringe_copy)):
                    yy = get_y2_from_shib(shib_rish, xs_c, ys_c, j)
                    xx += 1

                size_of_rish = TwoPointsDistance([xx,yy],[xs_c,ys_c])
                shib_zarib = len(fringe_copy) / size_of_rish
                img2 = cv2.resize(fringe_copy, (len(fringe_copy[0]), round(len(fringe_copy) * shib_zarib - perspective_i)),
                                  interpolation=cv2.INTER_CUBIC)
                xx = xs_c
                for j in range(xs_c, xs_c + len(img2)):
                    yy = get_y2_from_shib(shib_rish, xs_c, ys_c, j)
                    if ((xx < 0) or (yy >= img1h) or (xx >= img1w) or (yy < 0)): break
                    if (img2[j - xs_c][i - ys][3] == 255):
                        if (xx - 1 >= 0):
                            img1[yy2][xx] = img1[yy][xx - 1] = img1[yy2][xx - 1]
                        img1[yy][xx] = img2[j - xs_c][i - ys][0:3]
                    yy2 = yy
                    xx += 1
            else:
                dir_steps_y = sign(shib_rish)
                for j in range(xs_c, xs_c + len(fringe_copy)):
                    xx = get_x2_from_shib(shib_rish, xs_c, ys_c, yy)
                    yy += dir_steps_y

                # size_of_rish = math.sqrt(math.pow(xx - xs_c, 2) + math.pow(yy - ys_c, 2))
                size_of_rish = TwoPointsDistance([xx,yy],[xs_c,ys_c])
                shib_zarib = len(fringe_copy) / size_of_rish
                img2 = cv2.resize(fringe_copy, (len(fringe_copy[0]), round(len(fringe_copy) * shib_zarib- perspective_i)),
                                  interpolation=cv2.INTER_CUBIC)
                yy = i
                for j in range(xs_c, xs_c + len(img2)):
                    xx = get_x2_from_shib(shib_rish, xs_c, ys_c, yy)
                    if ((xx < 0) or (yy >= img1h) or (xx >= img1w) or (yy < 0)): break
                    if (img2[j - xs_c][i - ys][3] == 255):
                        if ((yy - dir_steps_y >= 0) and (yy - dir_steps_y < len(img1))):
                            img1[yy][xx2] = img1[yy - dir_steps_y][xx] = img1[yy - dir_steps_y][xx2]
                        img1[yy][xx] = img2[j - xs_c][i - ys][0:3]
                    xx2 = xx
                    yy += dir_steps_y
            shib_rish += shib_diff
            perspective_i += perspective_diff
            if (shib_left < shib_right and shib_rish > shib_right):
                shib_rish = shib_right
            elif (shib_left > shib_right and shib_rish < shib_right):
                shib_rish = shib_right

            if (i == ys + img2w - 1):
                xs = get_x2_from_shib(shib_khat, xs, ys, i)
                ys = i


def DrawFringeOnCarpet_X(img1, img2, x, y, x2, shib_left, shib_right, shib_khat, perspective_rate, nearest_vertex_23):
    xs, ys = x, y
    img1h, img1w, img2w = len(img1), len(img1[0]), len(img2[0])
    if (abs(shib_left) > 10 and abs(shib_right) < 1): shib_left = 10 * sign(shib_left)
    if (abs(shib_right) > 10 and abs(shib_left) < 1): shib_right = 10 * sign(shib_right)

    shib_rish = shib_left
    shib_diff = (shib_right - shib_left) / abs(x - x2)

    perspective_i = 0
    perspective_diff = perspective_rate * len(img2) / abs(x - x2)
    logger.Log('nearest_vertex_23: '+str(nearest_vertex_23))
    if (nearest_vertex_23 == 2):
        perspective_diff *= -1
        perspective_i = perspective_rate * len(img2)

    fringe_copy = deepcopy(img2)
    while (True):
        # draw fringe on carpet once
        for i in range(xs, xs + img2w):
            if (i > x2 or i > img1w):
                return img1
            yy = get_y2_from_shib(shib_khat, xs, ys, i)
            yy2 = yy
            xx = i
            xx2 = xx

            ys_c = yy
            xs_c = i
            if (shib_rish > 1 or shib_rish < -1):
                for j in range(ys, ys + len(fringe_copy)):

                    xx = get_x2_from_shib(shib_rish, i, ys, j)
                    yy += 1

                size_of_rish = TwoPointsDistance([xx, yy], [xs_c, ys_c])
                shib_zarib = len(fringe_copy) / size_of_rish
                img2 = cv2.resize(fringe_copy,
                                  (len(fringe_copy[0]), round(len(fringe_copy) * shib_zarib - perspective_i)),
                                  interpolation=cv2.INTER_CUBIC)
                yy = ys_c
                for j in range(ys, ys + len(img2)):
                    xx = get_x2_from_shib(shib_rish, i, ys, j)
                    if ((xx < 0) or (yy >= img1h) or (xx >= img1w) or (yy < 0)): break
                    if (img2[j - ys][i - xs][3] == 255):
                        img1[yy - 1][xx] = img1[yy - 1][xx2]
                        img1[yy][xx2] = img1[yy - 1][xx2]
                        img1[yy][xx] = img2[j - ys][i - xs][0:3]
                    xx2 = xx
                    yy += 1
            else:
                dir_steps_y = sign(shib_rish)
                for j in range(ys, ys + len(fringe_copy)):
                    yy = get_y2_from_shib(shib_rish, xs_c, ys_c, xx)
                    xx += dir_steps_y

                size_of_rish = TwoPointsDistance([xx, yy], [xs_c, ys_c])
                shib_zarib = len(fringe_copy) / size_of_rish
                img2 = cv2.resize(fringe_copy,
                                  (len(fringe_copy[0]), round(len(fringe_copy) * shib_zarib - perspective_i)),
                                  interpolation=cv2.INTER_CUBIC)
                xx = i
                for j in range(ys, ys + len(img2)):
                    yy = get_y2_from_shib(shib_rish, xs_c, ys_c, xx)
                    if ((xx < 0) or (yy >= img1h) or (xx >= img1w) or (yy < 0)): break
                    if (img2[j - ys][i - xs][3] == 255):
                        img1[yy][xx - 1 * dir_steps_y] = img1[yy2][xx - 1 * dir_steps_y]
                        img1[yy2][xx] = img1[yy2][xx - 1 * dir_steps_y]
                        img1[yy][xx] = img2[j - ys][i - xs][0:3]
                    yy2 = yy
                    xx += dir_steps_y

            shib_rish += shib_diff
            perspective_i += perspective_diff
            if (shib_left < shib_right and shib_rish > shib_right):
                shib_rish = shib_right
            elif (shib_left > shib_right and shib_rish < shib_right):
                shib_rish = shib_right

            if (i == xs + img2w - 1):
                ys = get_y2_from_shib(shib_khat, x, y, i)
                xs = i


def sign(num):
    if (num >= 0.0):
        return 1
    else:
        return -1


def get_x2_from_shib(shib, x1, y1, y2):
    return int(round(((y2 - y1) + shib * x1) / shib))


def get_y2_from_shib(shib, x1, y1, x2):
    return int(round(shib * (x2 - x1) + y1))

def CalculateSlope(p1,p2):
    epsilon = 0
    if(abs(p1[0] - p2[0]) < 0.1): epsilon = 0.01
    return (p1[1] - p2[1]) / (p1[0] - p2[0] + epsilon)

def rotate_points(x1, y1, x2, y2, x3, y3, x4, y4, rot_number, rows, cols):
    res = (x1, y1, x2, y2, x3, y3, x4, y4)
    if (rot_number == 1):
        res = (y1, cols - 1 - x1, y2, cols - 1 - x2, y3, cols - 1 - x3, y4, cols - 1 - x4)
    if (rot_number == -1):
        res = (rows - 1 - y1, x1, rows - 1 - y2, x2, rows - 1 - y3, x3, rows - 1 - y4, x4)
    if (abs(rot_number) == 2):
        res = (cols - 1 - x1, rows - 1 - y1, cols - 1 - x2, rows - 1 - y2, cols - 1 - x3, rows - 1 - y3, cols - 1 - x4,
               rows - 1 - y4)
    return res


def ConverImageToBase64(image):
    buff = BytesIO()
    image.save(buff, format="PNG")
    return "data:application/octet-stream;base64," + base64.b64encode(buff.getvalue()).decode("utf-8")

def DrawPointsOnPicture(Picture,Points,Show=True,Color=(0,255,0),Pen=10):
    pass
    # for i in Points:
    #     cv2.line(Picture, (i[0], i[1]), (i[0], i[1]),Color, Pen)
    # if(Show):
    #     ShowImage(Picture)
imgc = 0;
def ShowImage(Picture,BGR2RGB=True,cmap=""):
    global imgc
    Picture = np.float32(Picture)
    cv2.imwrite(str(imgc) + '.png', Picture)
    imgc += 1
    # pass
    # if(BGR2RGB):
    #     Picture = cv2.cvtColor(Picture, cv2.COLOR_BGR2RGB)
    # if(cmap==""):
    #     plt.imshow(Picture)
    # else:
    #     plt.imshow(Picture,cmap)
    # plt.show()

def FourPointTransform(Image, Points):
    rect = OrderPointsClockWise (Points)
    (tl, tr, br, bl) = rect
    widthA = TwoPointsDistance(br,bl)
    widthB = TwoPointsDistance(tr,tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = TwoPointsDistance(tr,br)
    heightB = TwoPointsDistance(tl,bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(Image, M, (maxWidth, maxHeight))

    return warped

def OrderPointsClockWise(Points):
    p = deepcopy(Points).tolist()
    top_point1 = FindSpecialPoints(p)[0]
    p.remove(top_point1)
    top_point2 = FindSpecialPoints(p)[0]
    bottom_point1 = FindSpecialPoints(p)[2]
    p.remove(bottom_point1)
    bottom_point2 = FindSpecialPoints(p)[2]

    if(top_point1[0] < top_point2[0]):
        tl = top_point1
        tr = top_point2
    else:
        tl = top_point2
        tr = top_point1

    if (bottom_point1[0] < bottom_point2[0]):
        bl = bottom_point1
        br = bottom_point2
    else:
        bl = bottom_point2
        br = bottom_point1

    # print("tl tr br bl : ",tl,tr,br,bl)
    return np.asarray([tl,tr,br,bl],dtype='float32')

def FindSpecialPoints(Points):
    most_top_point = Points[0]
    most_bottom_point = Points[0]
    most_right_point = Points[0]
    most_left_point = Points[0]

    for i in range(1, len(Points)):
        if (Points[i][0] > most_right_point[0]): most_right_point = Points[i]
        if (Points[i][0] < most_left_point[0]): most_left_point = Points[i]
        if (Points[i][1] > most_bottom_point[1]): most_bottom_point = Points[i]
        if (Points[i][1] < most_top_point[1]): most_top_point = Points[i]

    return (most_top_point,most_right_point,most_bottom_point,most_left_point)

def TwoPointsDistance(p1,p2):
    return round(math.sqrt(pow(p1[0] - p2[0],2) + pow(p1[1] - p2[1],2)))

def DetectCarpetCorners(img, corners):
    median = np.median(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
    sigma = 0.33
    lower_thresh = int(max(0, (1.0 - sigma) * median))
    upper_thresh = int(min(255, (1.0 + sigma) * median))
    edges = cv2.Canny(img, lower_thresh, upper_thresh)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=3)
    ret, labels = cv2.connectedComponents(edges, connectivity=8)

    colors_count = {}
    for i in range(0, len(edges)):
        for j in range(0, len(edges[0])):
            if (labels[i][j] != 0):
                if (labels[i][j] in colors_count.keys()):
                    colors_count[labels[i][j]] = colors_count[labels[i][j]] + 1
                else:
                    colors_count[labels[i][j]] = 1

    delete_bound = 2500
    list_accept_colors = []
    for i in colors_count.keys():
        if (colors_count[i] >= delete_bound):
            list_accept_colors.append(i)

    for i in range(0, len(edges)):
        for j in range(0, len(edges[0])):
            if (labels[i][j] in list_accept_colors):
                labels[i][j] = 255
            else:
                labels[i][j] = 0
    edges = labels
    ShowImage(edges,False,'gray')
    #---------------------------------------------------------------------------------------------------------------------------------
    plain_edges = GetPlainEdges(edges)
    plain_edges = np.float32(plain_edges)
    # ShowImage(plain_edges,False,'gray')
    countors = GetCountors(plain_edges)
    for i in range(0,len(countors)):
        for j in range(i, len(countors)):
            if(i == j):continue
            cv2.line(plain_edges, (countors[i][0],countors[i][1]), (countors[j][0],countors[j][1]), 0, 1)

    ShowImage(plain_edges, False, 'gray')
    plain_edges = ImageBitwiseNot(GetPlainEdges(ImageBitwiseNot(plain_edges)))
    ShowImage(plain_edges, False, 'gray')

    aruco_tl,aruco_tr,aruco_br,aruco_bl =   np.asarray(OrderPointsClockWise(corners),dtype='int32')

    aruco_top_edge_slope = CalculateSlope(aruco_tl,aruco_tr)
    aruco_right_edge_slope = CalculateSlope(aruco_tr, aruco_br)
    aruco_bottom_edge_slope = CalculateSlope(aruco_br, aruco_bl)
    aruco_left_edge_slope = CalculateSlope(aruco_tl, aruco_bl)

    firstHor = LastOccurWhitePoint(plain_edges,aruco_tr[0],aruco_tr[1],aruco_top_edge_slope)
    secondHor = LastOccurWhitePoint(plain_edges, aruco_br[0], aruco_br[1], aruco_bottom_edge_slope)
    firstVer = LastOccurWhitePoint(plain_edges, aruco_tr[0], aruco_tr[1], aruco_right_edge_slope)
    secondVer= LastOccurWhitePoint(plain_edges, aruco_tl[0], aruco_tl[1], aruco_left_edge_slope)

    edge_points = firstHor+secondHor+firstVer+secondVer
    DrawPointsOnPicture(img,edge_points)

    shib_right = CalculateSlope(firstHor[0],secondHor[0])
    shib_left   = CalculateSlope(firstHor[1],secondHor[1])
    shib_top    = CalculateSlope(firstVer[1],secondVer[1])
    shib_bottom = CalculateSlope(firstVer[0],secondVer[0])

    infinite_lines = np.array([[0 for j in range(len(img[0]))] for i in range(len(img))])

    line1 = DrawLineWithSlope(infinite_lines, firstVer[0][0],firstVer[0][1],shib_bottom)
    line2 = DrawLineWithSlope(infinite_lines, firstHor[0][0],firstHor[0][1], shib_right)
    line3 = DrawLineWithSlope(infinite_lines, firstVer[1][0],firstVer[1][1], shib_top)
    line4 = DrawLineWithSlope(infinite_lines, firstHor[1][0],firstHor[1][1], shib_left)

    rect_vertices = []

    rect_vertices.append(CalculateMinDistTwoLines(line1,line2)[0][0])
    rect_vertices.append(CalculateMinDistTwoLines(line2, line3)[0][0])
    rect_vertices.append(CalculateMinDistTwoLines(line3, line4)[0][0])
    rect_vertices.append(CalculateMinDistTwoLines(line1, line4)[0][0])

    ShowImage(infinite_lines,False,'gray')

    return rect_vertices


def ImageBitwiseNot(img):
    res = deepcopy(img)
    for i in range(0,len(img)):
        for j in range(0,len(img[0])):
            if(res[i][j] == 0):
                res[i][j] = 255
            else:
                res[i][j] = 0
    return res
def GetCountors(edges):
    img = np.float32(edges)
    countors = cv2.goodFeaturesToTrack(img, 1000, 0.01, 10)
    countors = np.int0(countors)
    points = []
    for r in countors:
        x, y = r.ravel()
        points.append([x,y])

    return points


def GetPlainEdges(edges):
    plain_edges_left = [[0 for j in range(0, len(edges[0]))] for i in range(0, len(edges))]
    plain_edges_right = [[0 for j in range(0, len(edges[0]))] for i in range(0, len(edges))]
    plain_edges_top = [[0 for j in range(0, len(edges[0]))] for i in range(0, len(edges))]
    plain_edges_bottom = [[0 for j in range(0, len(edges[0]))] for i in range(0, len(edges))]
    for i in range(0, len(edges[0])):
        for j in range(0, len(edges)):
            if (edges[j][i] == 255):
                break
            else:
                plain_edges_top[j][i] = 255
        for j in range(0, len(edges)):
            if (edges[len(edges) - j - 1][i] == 255):
                break
            else:
                plain_edges_bottom[len(edges) - j - 1][i] = 255

    for i in range(0, len(edges)):
        for j in range(0, len(edges[0])):
            if (edges[i][j] == 255):
                break
            else:
                plain_edges_left[i][j] = 255
        for j in range(0, len(edges[0])):
            if (edges[i][len(edges[0]) - j - 1] == 255):
                break
            else:
                plain_edges_right[i][len(edges[0]) - j - 1] = 255

    # ShowImage(plain_edges_bottom,False,'gray')
    # ShowImage(plain_edges_top, False, 'gray')
    # ShowImage(plain_edges_left, False, 'gray')
    # ShowImage(plain_edges_right, False, 'gray')

    plain_edges = plain_edges_bottom
    for i in range(len(edges)):
        for j in range(len(edges[0])):
            if (255 in [plain_edges_bottom[i][j], plain_edges_top[i][j], plain_edges_left[i][j],plain_edges_right[i][j]]):
                plain_edges[i][j] = 255

    return plain_edges

def CalculateMinDistTwoLines(l1,l2):
    dist = {}
    for i in l1:
        for j in l2:
            dist.update({(i,j):TwoPointsDistance([i[0],i[1]],[j[0],j[1]])})

    return [min(dist, key=dist.get),dist[min(dist, key=dist.get)]]

def DrawLineWithSlope(img,x,y,shib):
    line_points=[]
    if(abs(shib) <= 1):
        for i in range(x,len(img[0])):
            y2 = get_y2_from_shib(shib,x,y,i)
            if (IsPixelInPicture(img,[i,y2]) == False): break
            img[y2][i] += 1
            line_points.append((i,y2))
        for i in range(x-1,0,-1):
            y2 = get_y2_from_shib(shib,x,y,i)
            if (IsPixelInPicture(img, [i, y2]) == False): break
            img[y2][i] += 1
            line_points.append((i, y2))
    else:
        for i in range(y,len(img)):
            x2 = get_x2_from_shib(shib,x,y,i)
            if (IsPixelInPicture(img, [x2, i]) == False): break
            img[i][x2] += 1
            line_points.append((x2, i))
        for i in range(y-1,0,-1):
            x2 = get_x2_from_shib(shib,x,y,i)
            if (IsPixelInPicture(img, [x2, i]) == False): break
            img[i][x2] += 1
            line_points.append((x2, i))
    return line_points

def LastOccurWhitePoint(img,x,y,shib):
    result = []
    if(abs(shib) <= 1):
        for i in range(x,len(img[0])):
            y2 = get_y2_from_shib(shib,x,y,i)
            if (IsPixelInPicture(img,[i,y2]) == False): break
            if(img[y2][i] == 255):
                pixel_pointer = [i,y2]
        result.append(pixel_pointer)
        for i in range(x-1,0,-1):
            y2 = get_y2_from_shib(shib,x,y,i)
            if (IsPixelInPicture(img, [i, y2]) == False): break
            if (img[y2][i] == 255):
                pixel_pointer = [i, y2]
        result.append(pixel_pointer)
    else:
        for i in range(y,len(img)):
            x2 = get_x2_from_shib(shib,x,y,i)
            if (IsPixelInPicture(img, [x2, i]) == False): break
            if (img[i][x2] == 255):
                pixel_pointer = [x2, i]
        result.append(pixel_pointer)
        for i in range(y-1,0,-1):
            x2 = get_x2_from_shib(shib,x,y,i)
            if (IsPixelInPicture(img, [x2, i]) == False): break
            if (img[i][x2] == 255):
                pixel_pointer = [x2, i]
        result.append(pixel_pointer)
    return result
def DetectArucoCorners(img):
    filtered_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    aruco_parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, rejected_points = cv2.aruco.detectMarkers(filtered_frame, aruco_dictionary,parameters=aruco_parameters)

    if (len(corners) == 0):
        return []
    return np.array(corners, np.int32)[0][0]

def IsPixelInPicture(picture,p):
    if(p[0] < 0 or p[0] >= len(picture[0])):return False
    if (p[1] < 0 or p[1] >= len(picture)): return False
    return True

class DBManager:
    def ExecuteSqlQuery(self,query,isSelectQuery):

        cnxn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                              "Server=.;"
                              "Database=CarpetDB;"
                              "Trusted_Connection=yes;")


        cursor = cnxn.cursor()
        cursor.execute(query)
        if(isSelectQuery == False):
            cnxn.commit()

        #cnxn.close();
        return cursor;
    def insert_request(self,inputPic,outputPic,logs,ip):
        print(type(inputPic),type(outputPic))
        query  = "insert into Requests values('"+inputPic+"','"+outputPic+"',getdate(),'"+logs+"','"+ip+"')"
        print(query)
        self.ExecuteSqlQuery(query,False)


if __name__ == "__main__":
    app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024
    app.run()