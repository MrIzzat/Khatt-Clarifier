import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from tensorflow import keras
import tensorflow as tf
from Recognition.LoadDataset import max_len, num_to_char


model = tf.keras.Sequential()

def load_model():

    global model

    print(tf.__version__)

    model = tf.keras.models.load_model('./ICR_Model/epochs1000.h5')
    from hashlib import sha256

    input_ = str(model.get_weights())
    print(sha256(input_.encode('utf-8')).hexdigest())
    # model.load_weights('../ICR_Model/weights/epochs1000_2_weights')
    print(model.summary())


def predict(image):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.transpose(image)

    image = np.expand_dims(image, -1)

    preds = model.predict(np.array([image]),verbose=0)

    return preds




#model.load_weights('../ICR_Model/epochs1000_2_weights.h5')



def decode_batch_predictions(pred):
    global model


    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_len
    ]
    # Iterate over the results and get back the text.
    output_text = []
    #print(results)
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("UTF-8")
        output_text.append(res)
    #print(output_text)

    return output_text


def getCharacterAccuracy(test_ds):
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
test_model()