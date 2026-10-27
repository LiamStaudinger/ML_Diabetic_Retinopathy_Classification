import cv2
import numpy as np
import os

def preprocess(path, size=512, sigma=10):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > 7
    ys, xs = np.where(mask)
    img = img[ys.min():ys.max(), xs.min():xs.max()]      # crop
    img = cv2.resize(img, (size, size))                   # resize
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), sigma), -4, 128)  # normalize
    circle = np.zeros((size, size), np.uint8)
    cv2.circle(circle, (size//2, size//2), int(size*0.48), 1, -1)
    return img * circle[..., None]                         # mask corners

if __name__ == '__main__':
    train_dir = '/Users/liam/Desktop/CodeCoursework/CSC480/final_project/train_images'
    out_dir = '/Users/liam/Desktop/CodeCoursework/CSC480/final_project/processed_train_images'
    os.makedirs(out_dir, exist_ok=True)

    for image in os.listdir(train_dir):
        pre_processed_img = preprocess(os.path.join(train_dir, image))
        cv2.imwrite(os.path.join(out_dir, image), pre_processed_img)

    