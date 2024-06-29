import os
import cv2 as cv
from keras.src.layers import StringLookup

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from tensorflow import keras
import tensorflow as tf



model = tf.keras.Sequential()

characters = {'ن', 'س', 'ة', 'ك', '.', 'ء', '،', 'ف', 'ه', 'ب', 'خ', 'ؤ', 'غ', 'ز', 'ق', 'ث', 'ح', 'ذ', 'ئ', 'آ', 'إ', 'ض', 'ش', 'م', 'ر', 'ص', 'ا', 'ٍ', ':', 'و', 'أ', 'ظ', 'ل', 'ج', 'د', 'ع', 'ط', 'ت', 'ي'}
char_to_num = StringLookup(vocabulary=list(sorted(characters)), mask_token=None)

# Mapping integers back to original characters.
num_to_char = StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


def load_model():

    global model

    print(tf.__version__)

    model = tf.keras.models.load_model('./ICR_Model/epochs1000.h5')
    model = tf.keras.models.load_model('./ICR_Model/currBest_may_28_2024.h5')
    from hashlib import sha256

    input_ = str(model.get_weights())
    print(sha256(input_.encode('utf-8')).hexdigest())
    # model.load_weights('../ICR_Model/weights/epochs1000_2_weights')
    print(model.summary())
def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount ofpadding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image

def preprocess_image2(image, img_size=(64, 32)):
    image = tf.image.decode_jpeg(image, 1)

    image = distortion_free_resize(image, img_size)

    image =  tf.where(image > 80, 255, 0)#Basic thresholding

    image = tf.cast(image, tf.float32) / 255.0

    return image


def predict2(image):

    image = cv.imencode('.jpg', image)[1]
    image = preprocess_image2(image.tobytes())

    preds = model.predict(np.array([image]),verbose=0)

    return preds




#model.load_weights('../ICR_Model/epochs1000_2_weights.h5')



def decode_batch_predictions(pred):
    max_len=7
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=False)[0][0][
        :, :max_len
    ]
    # Iterate over the results and get back the text.
    output_text = []
    #print(results)
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("UTF-8")
        output_text.append(res)

    return output_text


def getCharacterAccuracy(test_ds):
    from Recognition.LoadDataset import max_len, num_to_char

    global model
    # for batch in test_ds.take(1):
    from Recognition.LoadDataset import max_len

    totalCharacters = 0
    wrongCharacters = 0
    batchNumber = 0

    for batch in test_ds:
        # batch = tuple(test_ds.take(1))[0]
        batchNumber += 1
        batch_images, batch_labels = batch["image"], batch["label"]
        preds = model.predict(batch_images)
        #pred_texts = decode_batch_predictions(preds)

        input_len = np.ones(preds.shape[0]) * preds.shape[1]
        # Use greedy search. For complex tasks, you can use beam search.
        results = keras.backend.ctc_decode(preds, input_length=input_len, greedy=True)[
                      0
                  ][0][:, :max_len]
        AllPredictions = []
        # All predicitons
        for res in results:
            res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
            res = tf.squeeze(res)
            norRes = tf.Variable(res)

            # print(norRes.shape)
            if len(norRes.shape) == 0:
                norRes = tf.stack((norRes, 99))
                while tf.size(norRes) < 7:
                    norRes = tf.concat((norRes, [99]), axis=-1)
            else:
                while tf.size(norRes) < 7:
                    norRes = tf.concat((norRes, [99]), axis=-1)

            AllPredictions.append(norRes)

        for i in range(len(AllPredictions)):
            #print(batch_labels[i])
            for j in range(7):
                # print(AllPredictions[i][j])
                # print(batch_labels[i][j])
                # print()

                if AllPredictions[i][j] != batch_labels[i][j]:
                    # print("Entered hell yeah")
                    wrongCharacters += 1
                totalCharacters += 1
                # print()
        else:
            continue

    print(wrongCharacters)
    print(totalCharacters)
    errorRate = (wrongCharacters / totalCharacters)
    print("Error Rate: ", errorRate * 100)

    accuracy = 1 - errorRate

    print("Accuracy: ", accuracy * 100)




def test_model():
    from Recognition import LoadDataset

    getCharacterAccuracy(LoadDataset.test_ds)

load_model()
#test_model()