import cv2 import tensorflow.lite as tflite import numpy as np
import time from keras import backend as K

def preprocess (img): 
    (h, w) img.shape
    final img np.ones([64, 256])*255
    # crop 
    if w > 256: 
        img img[:, :256]
    if h 64: 
        img img[:64, :]
    final_img[:h, :w] img 
    return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)

alphabets=u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-'"

def label_to_num(label):
    label num=[]
    for ch in label:
        label_num.append(alphabets.find(ch))
    return np.array(label_num)

def num to label (num):
    ret=""
    for ch in num:
        if ch==-1:
            break
        else:
            ret+=alphabets [ch]
return ret

interpreter = tf.lite.Interpreter(model_path="/content/drive/MyDrive/infernece/model1.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

test_dir = "/content/drive/MyDrive/mp_data/written_name_test_v2.csv"
test = pd.read_csv(test_dir)

tt=0

img_dir = "/content/drive/MyDrive/mp_data/test/"  + test.loc[5,'FILENAME']
image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE) 
plt.imshow(image, cmap='gray')
image = preprocess(image)
image = image/255.
input_shape = input_details[0]["shape"]
#print(input_shape)
input_data = np.array(image).reshape(1,256,64,1)

interpreter.set_tensor(input_details[0]["index"], input_data.astype(np.float32))

start=time.time()
interpreter.invoke()
end=time.time()
tt = tt + end-start
# print("time taken for inference: ",tt," sec")

output_data = interpreter.get_tensor(output_details[0]["index"])
# print(output_data.shape)
decoded = K.get_value(K.ctc_decode(output_data, input_length=np.ones(output_data.shape[0])*output_data.shape[1], 
                                    greedy=True)[0][0])
# print(num_to_label(decoded[0]))
plt.title(num_to_label(decoded[0]), fontsize=12)
plt.axis('off')

# plt.subplots_adjust(wspace=0.2, hspace=-0.8)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print('Avg Inference time: ',tt,'sec')
