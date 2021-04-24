import cv2
import numpy as np
import copy
#events = [i for i in dir(cv2) if 'EVENT' in i]
#print (events)

def put_text(image,txt,x0=50,y0=50):

    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    # org 
    org = (x0, y0) 
    # fontScale 
    fontScale = 1   
    # Blue color in BGR 
    color = (0, 255, 0) 
    # Line thickness of 2 px 
    thickness = 2
    # Using cv2.putText() method 
    image = cv2.putText(image, txt, org, font,fontScale, color, thickness, cv2.LINE_AA) 
    return image
  

class MousePts:
    def __init__(self,img,windowname='image'):
        self.windowname = windowname
        self.img1 = img.copy()
        self.img = self.img1.copy()
        cv2.namedWindow(windowname,cv2.WINDOW_NORMAL)
        cv2.imshow(windowname,img)
        self.curr_pt = []
        self.point   = []

    def select_point(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.point.append([x,y])
            #print(self.point)
            cv2.circle(self.img,(x,y),3,(0,255,0),2)
        elif event == cv2.EVENT_MOUSEMOVE:
            self.curr_pt = [x,y]
            #print(self.point)
    
    def draw(self,closed=False):
        if len(self.point)>0:
            polys=copy.deepcopy(self.point)
            polys.append(self.curr_pt)
            self.img = cv2.polylines(img=self.img, pts=np.array([polys]), isClosed=closed, color=(255,0,0), thickness=2)

    def drawRect(self):
        if len(self.point)>0:
            start_point = (self.point[0][0],self.point[0][1]) 
            end_point = (self.curr_pt[0],self.curr_pt[1]) 
            color = (255, 0, 0) 
            thickness = 2
            self.img = cv2.rectangle(self.img, start_point, end_point, color, thickness) 


    def getpt(self,count=1,img=None,plotline=True):
        if img is not None:
            self.img = img
            self.img1 = self.img.copy()
        else:
            self.img = self.img1.copy()
        
        txt = 'Click on '+str(count)+' points'
        self.img1 = put_text(self.img1,txt)
        cv2.namedWindow(self.windowname,cv2.WINDOW_NORMAL)
        cv2.imshow(self.windowname,self.img)
        cv2.setMouseCallback(self.windowname,self.select_point)
        self.point = []
        k=0
        while(1):
            self.img = self.img1.copy()
            self.draw()
            cv2.imshow(self.windowname,self.img)
            
            k = cv2.waitKey(20) & 0xFF
            if k == 27 or len(self.point)>=count:
                self.img = self.img1.copy()
                self.draw(closed=True)
                cv2.imshow(self.windowname,self.img)
                break
            #print(self.point)
        cv2.setMouseCallback(self.windowname, lambda *args : None)
        #cv2.destroyAllWindows()
        return self.point, self.img
        
    def selectRect(self,img,windowname='RectSelect'):

        self.img = img
        self.img1 = self.img.copy()
        self.windowname = windowname

        txt = 'Click on corner points, r to reset, enter to finish'
        self.img1 = put_text(self.img1,txt)
        cv2.namedWindow(self.windowname,cv2.WINDOW_NORMAL)
        cv2.imshow(self.windowname,self.img)
        cv2.setMouseCallback(self.windowname,self.select_point)
        self.point = []
        p1=[]
        p2=[]
        k=0
        while(1):
            self.img = self.img1.copy()
            self.drawRect()
            cv2.imshow(self.windowname,self.img)
            
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                return None,None
            if k == ord('r'):
                self.point = []

            if len(self.point)>1 and k==13:
                self.img = self.img1.copy()
                self.drawRect()
                cv2.imshow(self.windowname,self.img)
                p1, p2=self.point[0], self.curr_pt
                #import pdb;pdb.set_trace()
                x1,y1=min(p1[0],p2[0]),min(p1[1],p2[1])
                x2,y2=max(p1[0],p2[0]),max(p1[1],p2[1])
                p1=(x1,y1)
                p2=(x2,y2)
                break
                
            #print(self.point)
        cv2.setMouseCallback(self.windowname, lambda *args : None)
        #cv2.destroyAllWindows()
        return p1, p2

         
if __name__=='__main__':
    img = np.zeros((512,512,3), np.uint8)
    windowname = 'image'
    coordinateStore = MousePts(img, windowname)

    pts,img = MousePts(img, windowname).getpt(3)
    print(pts)
        
    pts,img = MousePts(img, windowname).getpt(3,img)
    print(pts)
    
    cv2.imshow(windowname,img)
    cv2.waitKey(0)
    

