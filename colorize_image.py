from argparse import ArgumentParser
import numpy as np
import cv2
import os
from utils import l_channel_from_lab_image, rgb_from_l_and_ab

# Parse input
parser = ArgumentParser()
parser.add_argument("--image", dest="image_path", required=True, help="Đường dẫn đến ảnh đầu vào")
args = parser.parse_args()
image_path = args.image_path

# Kiểm tra ảnh có tồn tại không
if not os.path.exists(image_path):
    print(f"Lỗi: Ảnh không tồn tại tại {image_path}")
    exit()

# Load ảnh
image = cv2.imread(image_path)
if image is None:
    print(f"Lỗi: Không thể đọc ảnh từ {image_path}")
    exit()

height, width = image.shape[:2]
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype("float32")

# Load model
weights_path = "models/colorization_release_v2.caffemodel"
config_path = "models/colorization_deploy_v2.prototxt"
if not os.path.exists(weights_path) or not os.path.exists(config_path):
    print("Lỗi: Không tìm thấy model .caffemodel hoặc .prototxt!")
    exit()

net = cv2.dnn.readNetFromCaffe(config_path, weights_path)

# Load cluster centers
pts = np.load("models/pts_in_hull.npy")
pts = pts.transpose().reshape(2, 313, 1, 1).astype("float32")

class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
net.getLayer(class8).blobs = [pts]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Chuẩn bị dữ liệu đầu vào
resized = cv2.resize(lab, (224, 224))
L = l_channel_from_lab_image(resized).astype("float32") - 50

# Dự đoán
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

# Kiểm tra kết quả
print(f"ab shape: {ab.shape}")  # Phải là (224, 224, 2)

# Resize kết quả về kích thước ảnh gốc
predicted_ab = cv2.resize(ab, (width, height), interpolation=cv2.INTER_CUBIC)
original_L = l_channel_from_lab_image(lab)
colorized = rgb_from_l_and_ab(original_L, predicted_ab)

# Hiển thị ảnh
cv2.imshow("Original", image)
cv2.imshow("Colorized", colorized)
cv2.waitKey(0)
cv2.destroyAllWindows()
