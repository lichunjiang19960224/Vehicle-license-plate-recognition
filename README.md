# Vehicle-license-plate-recognition

# You need library:
   
   tensorflow==1.9.0 or tensorflow-gpu==1.9.0
   
   numpy==1.16.0 and PIL.

# Train and predict
1.For province,digits and letters' training greatly, you can collect more dataset about province and put it into ./train_images/training-set/chinese-characters/, Similarly, digits and letters data can be puted into ./train_images/training-set/. When you have prepared the data, you can: python3 train_license_digits.py, train_license_letters.py and train_license_province.py.

2.In this project, I use a new method, using the gap between signal(white)  in binaryzation-image with black background, to segment license plate to province, letters and digits.

3.After training, you can run demo.py finish prediction. Of cause, you could use your own data.
