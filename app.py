from flask import Flask, render_template, request
from flask_cors import CORS
import cv2
import math
import pandas as pd
from binascii import a2b_base64
import numpy as np
import base64
from sklearn.mixture import GaussianMixture
import json
from shapely import geometry, ops
import keras




app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return render_template('index.html')



@app.route('/result', methods=['GET', 'POST'])
def result():
    string_data = request.get_data().decode('utf-8')
    string_data = string_data[23:]
    #print(string_data)
    binary_data = a2b_base64(string_data)
    fd = open('image.png', 'wb')
    fd.write(binary_data)
    fd.close()

    image=cv2.imread("image.png")
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    height, width = image.shape[:2]

    white_img = np.full((height,width, 3),
                        0, dtype = np.uint8)

    blur = cv2.GaussianBlur(gray,(5,5),0)
    bilateral = cv2.bilateralFilter(blur, 7, 50, 50)

    edges = cv2.Canny(bilateral,50,150,apertureSize=3)

    lines = cv2.HoughLinesP(
			edges, # Input edge image
			1, # Distance resolution in pixels
			np.pi/180, # Angle resolution in radians
			threshold=25, # Min number of votes for valid line
			minLineLength=10, # Min allowed length of line
			maxLineGap=15 # Max allowed gap between line for joining them
			)

    sumOfLengthsOfLines = 0
    # Iterate over points
    lines_list = []

    for points in lines:
        x1,y1,x2,y2=points[0]   
        sumOfLengthsOfLines =  sumOfLengthsOfLines + ((((x2 - x1 )**2) + ((y2-y1)**2))**0.5)

    avg = sumOfLengthsOfLines/len(lines)

    for points in lines:
        x1,y1,x2,y2=points[0]
        if ((((x2 - x1 )**2) + ((y2-y1)**2) )**0.5) >= width*0.08:
            cv2.line(image,(x1,y1),(x2,y2),(0,0,0),3)
            cv2.line(white_img,(x1,y1),(x2,y2),(255,255,255),6)

    image2=cv2.imread("image.png")

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    height, width = image.shape[:2]

    blur = cv2.GaussianBlur(gray,(5,5),0)
    bilateral = cv2.bilateralFilter(blur, 9, 50, 50)
    edges = cv2.Canny(bilateral,50,150,apertureSize=3)

    lines = cv2.HoughLinesP(
			edges, # Input edge image
			1, # Distance resolution in pixels
			np.pi/180, # Angle resolution in radians
			threshold=25, # Min number of votes for valid line
			minLineLength=10, # Min allowed length of line
			maxLineGap=10 # Max allowed gap between line for joining them
			)

    sumOfLengthsOfLines = 0
    # Iterate over points
    lines_list = []
    for points in lines:
        # Extracted points nested in the list
        x1,y1,x2,y2=points[0]
        #  Draw the lines joing the points
        sumOfLengthsOfLines =  sumOfLengthsOfLines + ((((x2 - x1 )**2) + ((y2-y1)**2))**0.5)
        # Maintain a simples lookup list for points

    avg = sumOfLengthsOfLines/len(lines)
    theta=[]
    linescsv=[]
    for points in lines:
        x1,y1,x2,y2=points[0]
        if ((((x2 - x1 )**2) + ((y2-y1)**2) )**0.5) >=width*0.08:
            cv2.line(image2,(x1,y1),(x2,y2),(0,255,0),1)
            linescsv.append(points[0])
            if x1==x2:
                theta.append(90)
            else:
                if(y1==y2):
                    temptheta=0
                else:
                    temptheta=math.atan((y2-y1)/(x2-x1))*(180/math.pi)
                if temptheta<0:
                    theta.append(temptheta+180)
                else:
                    theta.append(temptheta)

    _, im_arr = cv2.imencode('.jpg', image2)  # im_arr: image in Numpy one-dim array format.
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)

    data=pd.DataFrame()
    data['theta']=theta
    clust=10
    gmm = GaussianMixture(n_components=clust)
    gmm.fit(data)
    labels = gmm.predict(data)
    frame = pd.DataFrame(data)
    frame['cluster'] = labels

    x1=[]
    x2=[]
    y1=[]
    y2=[]
    c=[]
    for i in linescsv:
        x1.append(i[0])
        x2.append(i[2])
        y1.append(i[1])
        y2.append(i[3])
    for i in range(len(theta)):
        if theta[i]==90:
            c.append(0);
        elif theta[i]==0:
            c.append(y1[i])
        else:
            tan1=math.tan(theta[i]*math.pi/180)
            c.append(y1[i]-tan1*x1[i])

    
    df=pd.DataFrame()
    df['x1']=x1
    df['y1']=y1
    df['x2']=x2
    df['y2']=y2
    df['c']=c

    df['label']=labels
    df['theta']=theta
    xavg=[]
    yavg=[]
    k=len(labels)
    for i in range(k): 
        xavg.append((x1[i]+x2[i])/2)
        yavg.append((y1[i]+y2[i])/2)
    df['xavg']=xavg
    df['yavg']=yavg

    df.set_index(df.columns[5], inplace = True)
    df.sort_values(by=['label'],inplace=True)

    linespacing=[]
    for i in range(clust):
        if isinstance(df.xavg[i], np.floating)==False:
            close_points=[]
            point=[]
            len_point=len(df.xavg[i].values)
            for j in range(len_point):
                point.append([df.xavg[i].values[j],df.yavg[i].values[j],df.theta[i].values[j],df.c[i].values[j]])
            thetavg=sum(df.theta[i].values)/len(df.theta[i].values)
        
            if thetavg>=88 and thetavg<=92:
                point.sort(key=lambda x:x[0])
            else :
                point.sort(key=lambda x:x[3])
            
            for x in range(len_point-1):
                tan1=math.tan(point[x][2]*math.pi/180)
                tan2=math.tan(point[x+1][2]*math.pi/180)
                c1=(point[x][1]-tan1*point[x][0])
                c2=(point[x+1][1]-tan2*point[x+1][0])
                if 85<thetavg and thetavg<95:
                    dist=abs(point[x][0]-point[x+1][0])                                                           
                else:
                    dist=abs((c1-c2)/math.sqrt(((tan1+tan2)/2)**2+1))
                if dist<width/15:                                  #####  Set the value 
                
                    close_points.append([x,x+1])


            lst = [-1]*len_point
            lst_minpoint=[1]*len_point
#           print(i,close_points)
            for j in range(len(close_points)):                          ## list containing line which are far apart
#               print(close_points[j][0],close_points[j][1])
                if lst[close_points[j][0]]==-1 and lst[close_points[j][1]]==-1:
                    lst[close_points[j][0]]=1
                    lst[close_points[j][1]]=1
                    lst_minpoint[close_points[j][0]]=-1
                else:
                    lst[close_points[j][0]]=1  
                    lst[close_points[j][1]]=1
            spacing=0
            prevtan=-1
            prevc=-1
            prevx=-1
            prevy=-1
            size=0;
            for t in range(len_point):            
                if lst_minpoint[t]==-1 or lst[t]==-1:
                    size=size+1
                    if 85<thetavg and thetavg<95:
                        if(prevx!=-1):
                            spacing=spacing+point[t][0]-prevx 
                        prevx=point[t][0]                  
                    elif (thetavg>0 and thetavg<5) or (thetavg>175 and thetavg<180) :
                        if(prevy!=-1):
                            spacing=spacing+abs(point[t][1]-prevy)
                        prevy=point[t][1]
                    else:                  
                        tan1=math.tan(point[t][2]*math.pi/180)
                        c1=(point[t][1]-tan1*point[t][0])
                        if prevtan!=-1:
                            spacing=spacing+abs((c1-prevc)/math.sqrt(((tan1+prevtan)/2)**2+1))
                        
                        prevc=c1
                        prevtan=tan1
            linespacing.append(spacing)

    # RQD CALCULATION
    intersection=[]
    rqd=0
    nonzerodist=10
    for i in range(10):
        x_point=(width/10)*i
        Rqdlist=[]
        for j in range(len(x1)):
            if (x_point>=x1[j] and x_point<=x2[j]) or (x_point<=x1[j] and x_point>=x2[j]):
                if theta[i]!=90:
                    y_point=math.tan(theta[i]*math.pi/180)*x_point+c[i]
                    intersection.append([x_point,y_point])
        intersection.sort(key=lambda x:x[1])
        limit_dist=0
        dist=0
        for i in range(len(intersection)-1):
            tempdist=math.sqrt((intersection[i][1]-intersection[i+1][1])**2+(intersection[i][0]-intersection[i+1][0])**2)
            if(tempdist<height/2):
                limit_dist=limit_dist+tempdist
            dist=dist+tempdist
        if dist==0:
            nonzerodist=nonzerodist-1
        else:
            rqd=rqd+(dist-limit_dist)/dist
    RQD=rqd/nonzerodist

    #print(im_b64)
    res_uri = str(im_b64)
    res_uri = res_uri[2:]
    res_uri = res_uri[:-1]

    res = {'res_uri': res_uri, 'linespacing':linespacing, 'rqd': RQD}

    return json.dumps(res)


@app.route('/failure', methods=['GET', 'POST'])
def failure():
    string_data = request.get_data().decode('utf-8')
    arr = string_data.split();
    #print(arr)
    d_strike = int(arr[0])
    s_strike = int(arr[1])
    d_dip = int(arr[2])
    s_dip = int(arr[3])
    f_angle = int(arr[4])
    #print(d_strike)
    #print(type(d_strike))

    def pole2plunge_bearing(strike, dip):
    
        strike, dip = np.atleast_1d(strike, dip)
        bearing = strike - 90
        plunge = 90 - dip
        bearing[bearing < 0] += 360
        return plunge, bearing

    

    

    def sph2cart(lon, lat):
        x = np.cos(lat)*np.cos(lon)
        y = np.cos(lat)*np.sin(lon)
        z = np.sin(lat)
        return x, y, z
    
    def cart2sph(x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        lat = np.arcsin(z/r)
        lon = np.arctan2(y, x)
        return lon, lat

    def _rotate(lon, lat, theta, axis='x'):
        # Convert input to numpy arrays in radians
        lon, lat = np.atleast_1d(lon, lat)
        lon, lat = map(np.radians, [lon, lat])
        theta = np.radians(theta)

        # Convert to cartesian coords for the rotation
        x, y, z = sph2cart(lon, lat)

        lookup = {'x':_rotate_x, 'y':_rotate_y, 'z':_rotate_z}
        X, Y, Z = lookup[axis](x, y, z, theta)

        # Now convert back to spherical coords (longitude and latitude, ignore R)
        lon, lat = cart2sph(X,Y,Z)
        return lon, lat # in radians!

    def _rotate_x(x, y, z, theta):
        X = x
        Y = y*np.cos(theta) + z*np.sin(theta)
        Z = -y*np.sin(theta) + z*np.cos(theta)
        return X, Y, Z

    def _rotate_y(x, y, z, theta):
        X = x*np.cos(theta) + -z*np.sin(theta)
        Y = y
        Z = x*np.sin(theta) + z*np.cos(theta)
        return X, Y, Z

    def _rotate_z(x, y, z, theta):
        X = x*np.cos(theta) + -y*np.sin(theta)
        Y = x*np.sin(theta) + y*np.cos(theta)
        Z = z
        return X, Y, Z

    def antipode(lon, lat):
        x, y, z = sph2cart(lon, lat)
        return cart2sph(-x, -y, -z)
    
    def line(plunge, bearing):
        plunge, bearing = np.atleast_1d(plunge, bearing)
        # Plot the approriate point for a bearing of 0 and rotate it
        lat = 90 - plunge
        lon = 0
        lon, lat = _rotate(lon, lat, bearing)
        return lon, lat

    

    

    def pole(strike, dip):
        strike, dip = np.atleast_1d(strike, dip)
        mask = dip > 90
        dip[mask] = 180 - dip[mask]
        strike[mask] += 180
        # Plot the approriate point for a strike of 0 and rotate it
        lon, lat = -dip, 0.0
        lon, lat = _rotate(lon, lat, strike)
        return lon, lat



    class PlanarSliding(object):

        def __init__(self, strike, dip, fric_angle=35, latlim=20):
        
            self.strike = strike
            self.dip = dip
            self.fric_angle = fric_angle
            self.latlim = latlim
        
            if latlim <= 0 or latlim >= 90:
                raise ValueError('latlim must be > 0 and < 90')
            
            if dip <= 0 or dip > 90:
                raise ValueError('dip must be > 0 and <= 90')
            
            if dip <= fric_angle:
                raise ValueError('No planar sliding zones generated as the input'
                             ' slope dip is shallower than the friction angle')
        def check_failure(self, strikes, dips, curved_lateral_limits=True):
            strikes = (strikes-self.strike)%360
            dipdirs = (strikes+90)%360
        
            if curved_lateral_limits:
#               lons, lats = stereonet_math.pole(strikes, dips)
                lons, lats =pole(strikes, dips)
                lats = np.degrees(lats)
                within_lat = ((lats >= -self.latlim-1e-8) &  # with tolerance
                            (lats <= self.latlim+1e-8))
            else:
                within_lat = ((dipdirs >= 90-self.latlim) &
                            (dipdirs <= 90+self.latlim))

#          llons, llats = stereonet_math.line(dips, dipdirs)
            llons, llats =line(plunges, bearings)
            llons = np.degrees(llons)
            daylight = llons >= 90-self.dip-1e-8  # with tolerance
        
            fric_slip = dips >= self.fric_angle
        
            main = within_lat & fric_slip & daylight
            secondary = ~within_lat & fric_slip & daylight
        
            return main, secondary
    plunges,bearings=pole2plunge_bearing(d_strike,d_dip)
    p2=PlanarSliding(s_strike,s_dip,f_angle)
    tup = p2.check_failure(d_strike,d_dip);

    a = tup[0][0]
    print(tup)
    b = tup[1][0]
    str1 = "error"
    str2 = "error"
    if a:
        str1 = "true"
    else:
        str1 = "false"

    return str1


@app.route('/gsi', methods=['GET', 'POST'])
def gsi():
    string_data = request.get_data().decode('utf-8')
    string_data = string_data[23:]
    #print(string_data)
    binary_data = a2b_base64(string_data)
    fd = open('gsi.png', 'wb')
    fd.write(binary_data)
    fd.close()
    vgg_best_model = keras.models.load_model('vgg_16_-saved-model-60-acc-0.67.hdf5')
    img = cv2.resize(cv2.imread('gsi.png'),(150,150))
    img_normalized = img/255
    vgg16_image_prediction = np.argmax(vgg_best_model.predict(np.array([img_normalized])))
    str1 = str(vgg16_image_prediction*10)
    str2 = str(vgg16_image_prediction*10+10)
    ans = str1 + "-" + str2
    print(ans)
    return ans



if __name__ == "__main__":
    app.run()
