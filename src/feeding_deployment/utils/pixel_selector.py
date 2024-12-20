import cv2

class PixelSelector:
    def __init__(self):
        pass

    def load_image(self, img):
        self.img = img

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.clicks.append([x, y])
            cv2.circle(self.img, (x, y), 3, (200, 0, 0), -1)
            cv2.imshow("pixel_selector", self.img)

    def run(self, img, num_clicks=1):
        self.load_image(img)
        self.clicks = []
        cv2.namedWindow('pixel_selector')
        cv2.setMouseCallback('pixel_selector', self.mouse_callback)
        while True:
            cv2.imshow("pixel_selector", self.img)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:  # Escape key
                break
            if len(self.clicks) >= num_clicks:
                break
        return self.clicks