import cv2  #gerekli kütüphaneleri tanımlarız
import os 
from glob import glob
# eğer jpeg-->jpg yapmak istiyorsan '*.jpeg' yapmalısın ve 'j[:-4]' yapmalısın

png = glob('D:/ImageProcessing/YOLOv4/custom_yolo_model/dataset/spot_images/*.png')  #png uzantılı tüm dosyaları okuruz 

for j in png:  #png uzantılı resimler içinde geziniyoruz
    print(j)
    img = cv2.imread(j) 
    cv2.imwrite(j[:-3]+'jpg',img) #okuduğum resmin son üç karakterini al jpg ekle ve tekrardan img değişkenine eşitle
    os.remove(j) #ve ilk resmimi siliyorum