import cv2
import numpy as np
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

class ROISelector:
    def __init__(self, image_path):
        self.original_image = cv2.imread(image_path)  
        self.image = self.original_image.copy()
        self.clone = self.image.copy()       
        self.clone_temp = self.clone.copy()  
        self.image_flag = 0                  
        self.roi = np.zeros(self.image.shape[:2], dtype=np.uint8)  
        self.new_roi = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.drawing = False   
        self.resizing = False  
        self.roi_selected = False      
        self.selected_handle = None    
        self.handle_size = 10          
        self.mode = 'rectangle' 
        self.operation = 'add'  
        self.start_point = None  
        self.rx=0; self.ry=0; self.rw=0; self.rh=0; 
        self.cx=0; self.cy=0; self.radius=0; 
        
    def find_selected_handle_circle(self, x, y, cx, cy, radius):
        """ 鏌ユ壘琚€変腑鐨勬帶鍒剁偣 """
        if abs(x - cx) <= self.handle_size and abs(y - cy) <= self.handle_size:
            return "center"
        elif abs((x - cx)) <= self.handle_size and abs((y - (cy - radius))) <= self.handle_size:
            return "top"
        elif abs((x - cx)) <= self.handle_size and abs((y - (cy + radius))) <= self.handle_size:
            return "bottom"
        elif abs((x - (cx - radius))) <= self.handle_size and abs((y - cy)) <= self.handle_size:
            return "left"
        elif abs((x - (cx + radius))) <= self.handle_size and abs((y - cy)) <= self.handle_size:
            return "right"
        else:
            return None
        
    def find_selected_handle_rectangle(self, x, y, rect):
        """ 鏌ユ壘琚€変腑鐨勮竟鎴栬 """
        rx, ry, rw, rh = rect
        if abs(x - rx) <= self.handle_size and abs(y - ry) <= self.handle_size:
            return "topleft"
        elif abs(x - (rx + rw)) <= self.handle_size and abs(y - ry) <= self.handle_size:
            return "topright"
        elif abs(x - rx) <= self.handle_size and abs(y - (ry + rh)) <= self.handle_size:
            return "bottomleft"
        elif abs(x - (rx + rw)) <= self.handle_size and abs(y - (ry + rh)) <= self.handle_size:
            return "bottomright"
        else:
            return None
        
    def update_circle(self, x, y, handle):
        """ 鏍规嵁鎺у埗鐐硅皟鏁村渾 """
        if handle == "center":
            self.cx, self.cy = x, y
        elif handle == "top":
            self.radius = abs(y - self.cy)
        elif handle == "bottom":
            self.radius = abs(y - self.cy)
        elif handle == "left":
            self.radius = abs(x - self.cx)
        elif handle == "right":
            self.radius = abs(x - self.cx)
            
    def update_rectangle(self, x, y, handle):
        """ 鏍规嵁杈圭紭璋冩暣 ROI """
        if handle == "topleft":
            self.rw, self.rh = self.rw + (self.rx - x), self.rh + (self.ry - y)
            self.rx, self.ry = x, y
            # self.rw, self.rh = self.rw + (self.start_point[0] - x), self.rh + (self.start_point[1] - y)
        elif handle == "topright":
            self.rw, self.rh = x - self.rx, self.rh + (self.ry - y)
            self.ry = y
            # self.rw, self.rh = x - self.rx, self.rh + (self.start_point[1] - y)
        elif handle == "bottomleft":
            self.rw, self.rh = self.rw + (self.rx - x), y - self.ry
            self.rx = x
            # self.rw, self.rh = self.rw + (self.start_point[0] - x), y - self.ry
        elif handle == "bottomright":
            self.rw, self.rh = x - self.rx, y - self.ry
            
    def draw_circle(self):
        """ 缁樺埗鍦嗗拰鎺у埗鐐� """
        # 鍦ㄥ摢涓敾甯冧笂锛燂紵锛燂紵锛燂紵  搴旇鍦╡vent涓婂睍绀�
        cv2.circle(self.image, (self.cx, self.cy), self.radius, (0, 255, 0), 2)
        cv2.circle(self.image, (self.cx, self.cy), self.handle_size, (255, 0, 0), -1)
        cv2.circle(self.image, (self.cx, self.cy - self.radius), self.handle_size, (255, 0, 0), -1)
        cv2.circle(self.image, (self.cx, self.cy + self.radius), self.handle_size, (255, 0, 0), -1)
        cv2.circle(self.image, (self.cx - self.radius, self.cy), self.handle_size, (255, 0, 0), -1)
        cv2.circle(self.image, (self.cx + self.radius, self.cy), self.handle_size, (255, 0, 0), -1)
        
    def draw_rectangle(self):
        """ 缁樺埗 ROI 鍜屾帶鍒剁偣 """
        # 鍦ㄥ摢涓敾甯冧笂锛燂紵锛燂紵锛燂紵
        cv2.rectangle(self.image, (self.rx, self.ry), (self.rx + self.rw, self.ry + self.rh), (0, 255, 0), 2)
        cv2.circle(self.image, (self.rx, self.ry), self.handle_size, (255, 0, 0), -1)
        cv2.circle(self.image, (self.rx + self.rw, self.ry), self.handle_size, (255, 0, 0), -1)
        cv2.circle(self.image, (self.rx, self.ry + self.rh), self.handle_size, (255, 0, 0), -1)
        cv2.circle(self.image, (self.rx + self.rw, self.ry + self.rh), self.handle_size, (255, 0, 0), -1)
        
    def display_roi(self):
        if self.image_flag == 0:
            if self.mode == 'circle':
                self.draw_circle()
            elif self.mode == 'rectangle':
                self.draw_rectangle()
        elif self.image_flag == 1:
            self.image = self.clone.copy()

    def draw_roi(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.roi_selected:
                self.drawing = True
                self.image_flag = 0   #######
                self.start_point = (x, y)
                if self.mode == 'circle':
                    self.cx, self.cy, self.radius = self.start_point[0], self.start_point[1], 0
                elif self.mode == 'rectangle':
                    self.rx, self.ry, self.rw, self.rh = self.start_point[0], self.start_point[1], 0, 0
            else:
                if self.mode == 'circle':
                    self.selected_handle = self.find_selected_handle_circle(
                        x, y, self.cx, self.cy, self.radius)
                    if self.selected_handle:
                        self.resizing = True
                        self.start_point = (x, y)
                elif self.mode == 'rectangle':
                    self.selected_handle = self.find_selected_handle_rectangle(
                        x, y, (self.rx, self.ry, self.rw, self.rh))
                    if self.selected_handle:
                        self.resizing = True
                        self.start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                if self.mode == 'circle':
                    self.radius = int(np.sqrt((self.start_point[0] - x) ** 2 + (self.start_point[1] - y) ** 2))
                elif self.mode == 'rectangle':
                    self.rx, self.ry = min(self.start_point[0], x), min(self.start_point[1], y)
                    self.rw, self.rh = abs(self.start_point[0] - x), abs(self.start_point[1] - y)
                # self.update_display(x, y)    #   杩欓噷瑕佷慨鏀�
            elif self.resizing:
                if self.mode == 'circle':
                    self.update_circle(x, y, self.selected_handle)
                elif self.mode == 'rectangle':
                    self.update_rectangle(x, y, self.selected_handle)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                self.roi_selected = True
            elif self.resizing:
                self.resizing = False

    def update_display(self):
        self.clone = self.clone_temp.copy()
        self.clone = cv2.addWeighted(self.clone, 1, cv2.cvtColor(self.roi, cv2.COLOR_GRAY2BGR), 0.5, 0)
        self.image_flag = 1
        
    def load_roi(self):
        self.new_roi = np.zeros(self.image.shape[:2], dtype=np.uint8)
        if self.mode == 'rectangle':
            cv2.rectangle(self.new_roi, (self.rx, self.ry), (self.rx + self.rw, self.ry + self.rh), 255, -1)
        elif self.mode == 'circle':
            cv2.circle(self.new_roi, (self.cx, self.cy), self.radius, 255, -1)

    def apply_operation(self):
        self.load_roi()
        if self.operation == 'add':
            self.roi = cv2.bitwise_or(self.roi, self.new_roi)
        elif self.operation == 'subtract':
            self.roi = cv2.bitwise_and(self.roi, cv2.bitwise_not(self.new_roi))
        self.update_display()
        self.drawing = False   # 鏄惁姝ｅ湪缁樺埗鎴栬皟鏁�
        self.resizing = False  # 鏄惁姝ｅ湪璋冩暣
        self.roi_selected = False      # 鏄惁閫夋嫨浜哛OI
        self.selected_handle = None    # 閫変腑鐨勬帶鍒剁偣
        self.start_point = None  # 鍒濆鐐�
        self.rx=0; self.ry=0; self.rw=0; self.rh=0; 
        self.cx=0; self.cy=0; self.radius=0
        

    def run(self):
        cv2.namedWindow("ROI Selector")
        cv2.setMouseCallback("ROI Selector", self.draw_roi)

        while True:
            self.image = self.original_image.copy()
            self.display_roi()
            cv2.imshow("ROI Selector", self.image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('a'):
                self.operation = 'add'
            elif key == ord('s'):
                self.operation = 'subtract'
            elif key == ord('c'):
                self.mode = 'circle'
            elif key == ord('r'):
                self.mode = 'rectangle'
            elif key == ord('f'):
                self.apply_operation()
            elif key == ord('q'):
                break
        cv2.destroyAllWindows()
        return self.roi

def ROI_bmp():
    Tk().withdraw()  # 闅愯棌鏍� Tkinter 绐楀彛
    image_path = askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.bmp;*.jpg;*.jpeg;*.png;*.tiff;*.tif")]
    )   # 瑕佹眰鐢ㄦ埛閫夋嫨涓€涓浘鍍忔枃浠�
    if not image_path:
        print("No image selected. Exiting.")
        return
    # 瀹炰緥鍖朢OISelector
    selector = ROISelector(image_path)
    final_roi = selector.run()
    # 灏� ROI 淇濆瓨鍒板綋鍓嶅伐浣滅洰褰�
    cv2.imwrite('final_roi.png', final_roi)
    print(f"Final ROI saved as 'final_roi.png' in the current directory.")
    # 灏哛OI淇濆瓨鍒版墍閫夊浘鍍忕殑鐩綍涓�
    image_dir = os.path.dirname(image_path)
    save_path = os.path.join(image_dir, 'roi.bmp')
    cv2.imwrite(save_path, final_roi)
    print(f"Final ROI saved as 'roi.bmp' in the directory of the selected image.")

if __name__ == "__main__":
    ROI_bmp()
