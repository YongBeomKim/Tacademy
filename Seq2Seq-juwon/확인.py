from scipy.misc import imread, imsave, imresize
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
img = imread('코끼리_tinted.jpg')
img1 = imread('햄스터_tinted.jpg')
img2 = imread('코끼리11.jpg')
img3 = imread('코끼리22.jpg')
img4 = imread('코끼리33.jpg')
img5 = imread('코끼리44.jpg')

data = [img, img1,img2,img3,img4]
char_arr = [c for c in 'SEP가아나기다코구끼라리와마오리바햄스사터뒤에물통 ']
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)+1
max = 10 #입력과 출력의 최대 글자 수


def attend(contexts, output):
    """ Attention Mechanism. """

    reshaped_contexts = tf.reshape(contexts, [-1, 256])



        # use 1 fc layer to attend
    logits1 =tf.layers.dense(reshaped_contexts,
                                units=1,
                                activation=None,
                                use_bias=False,
                                name='fc_a',reuse=tf.AUTO_REUSE)

    logits1 = tf.reshape(logits1, [-1, 196])

    logits2 = tf.layers.dense(output,units=196,activation=None,use_bias=False,name='fc_b',reuse=tf.AUTO_REUSE)

    logit = logits1 + logits2
    logit = tf.nn.softmax(logit)



    return logit

def conv2d(
           inputs,
           filters,
           kernel_size=(3, 3),
           strides=(1, 1),
           activation=tf.nn.relu,
           use_bias=True,
           name=None):

    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        activation=activation,
        use_bias=use_bias,
        name=name)


def max_pool2d(
               inputs,
               pool_size=(2, 2),
               strides=(2, 2),
               name=None):
    """ 2D Max Pooling layer. """
    return tf.layers.max_pooling2d(
        inputs=inputs,
        pool_size=pool_size,
        strides=strides,
        padding='same',
        name=name)


def dense(
          inputs,
          units,
          activation=tf.tanh,
          use_bias=True,
          name=None):
    """ Fully-connected layer. """

    return tf.layers.dense(
        inputs=inputs,
        units=units,
        activation=activation,
        use_bias=use_bias,
        name=name
        )

data=np.array(data)


###사진의 크기 224*224
images = tf.placeholder(
        dtype=tf.float32,
        shape=[None,224,224,3])

def build_vgg16(images):
    """ Build the VGG16 net. """

    conv1_1_feats = conv2d(images, 64, name='conv1_1')
    conv1_2_feats = conv2d(conv1_1_feats, 64, name='conv1_2')
    pool1_feats = max_pool2d(conv1_2_feats, name='pool1')

    conv2_1_feats =conv2d(pool1_feats, 128, name='conv2_1')
    conv2_2_feats = conv2d(conv2_1_feats, 128, name='conv2_2')
    pool2_feats = max_pool2d(conv2_2_feats, name='pool2')

    conv3_1_feats = conv2d(pool2_feats, 256, name='conv3_1')
    conv3_2_feats = conv2d(conv3_1_feats, 256, name='conv3_2')
    conv3_3_feats = conv2d(conv3_2_feats, 256, name='conv3_3')
    pool3_feats = max_pool2d(conv3_3_feats, name='pool3')

    conv4_1_feats = conv2d(pool3_feats, 256, name='conv4_1')
    conv4_2_feats = conv2d(conv4_1_feats, 256, name='conv4_2')
    conv4_3_feats = conv2d(conv4_2_feats, 256, name='conv4_3')
    pool4_feats = max_pool2d(conv4_3_feats, name='pool4')

    conv5_1_feats = conv2d(pool4_feats, 256, name='conv5_1')
    conv5_2_feats = conv2d(conv5_1_feats, 256, name='conv5_2')
    conv5_3_feats = conv2d(conv5_2_feats, 256, name='conv5_3')#14*14*256 으로 출력

    reshaped_conv5_3_feats = tf.reshape(conv5_3_feats,
                                        [-1, 196, 256])


    context_mean = tf.reduce_mean(reshaped_conv5_3_feats, axis=1)

    memory = dense(context_mean,
                           units=4096,
                           activation=None,
                           name='fc_a')
    output = dense(context_mean,
                           units=4096,
                           activation=None,
                           name='fc_b')

    return reshaped_conv5_3_feats, memory, output

ima, state, out = build_vgg16(images)
label = [['','코끼리와 오리'],['','햄스터와 물통'],['','코끼리'],['','코끼리'],['','코끼리']]##입력데이터는 필요없기에 비워둠.



def make_batch(seq_data):

    output_batch = []
    target_batch = []

    for seq in seq_data:


        # 디코더 셀의 입력값. 시작을 나타내는 S 심볼을 맨 앞에 붙여준다.

        output = [num_dic[n] for n in ('S' + seq[1])]

        # 학습을 위해 비교할 디코더 셀의 출력값. 끝나는 것을 알려주기 위해 마지막에 E 를 붙인다.
        target = [num_dic[n] for n in (seq[1] + 'E')]

        ##아래는 글자의 최대수보다 낮으면 패딩으로 채워주고 크면 최대만큼 잘라냄
        if len(output) <= max:
            for i in range(max - len(output)):
                output.append(2)
        else:
            output = output[:max]

        if len(target) <= max:
            for i in range(max - len(target)):
                target.append(2)
        else:
            target = target[:max]

        output_batch.append(np.eye(dic_len)[output])

        target_batch.append(target)

    return  output_batch, target_batch


learning_rate = 0.0001
n_hidden = 4096 #CNN출력의 사이즈와 맞춰줌
total_epoch = 1001

n_class =dic_len
n_input =dic_len
output_batch, target_batch = make_batch(label)


dec_input = tf.placeholder(tf.float32, [None, None, n_input])
# [batch size, time steps]

targets = tf.placeholder(tf.int64, [None, None])
output_batch = np.array(output_batch)

with tf.variable_scope('decode'):
    dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)

    '''Seq2Seq 모델에서는 인코더 셀의 최종 상태값을 디코더 셀의 초기 상태값으로 넣어주지만 이 모델에서는 CNN에서 나온 출력값을 초기상태값으로 넣어줌.'''

    predictions = []
    sente = []
    al = []
    for i in range(max):
        if i == 0:
            dec_inputa = dec_input[:,i,:]
            alpha=attend(ima,out)

            context = tf.reduce_sum(ima * tf.expand_dims(alpha, 2), axis=1)

            a = tf.nn.softmax(alpha)
            a = tf.reshape(a, [14, 14])
            al.append(a)




            current_input = tf.concat([context, dec_inputa], 1)

            outputs, dec_states = dec_cell(current_input, state)
            expanded_output = tf.concat([outputs, context, dec_inputa], axis=1)
            model = tf.layers.dense(expanded_output, n_class, activation=None)  ##단어 갯수만큼 출력을 맞춰줌
            probs = tf.nn.softmax(model)
            prediction = tf.argmax(model, 1)
            sente.append(model)
            predictions.append(prediction)

        else:
            dec_inputa = dec_input[:, i, :]
            alpha = attend(ima, outputs)
            context = tf.reduce_sum(ima * tf.expand_dims(alpha, 2), axis=1)

            a = tf.nn.softmax(alpha)
            a = tf.reshape(a, [14, 14])
            al.append(a)

            current_input = tf.concat([context, dec_inputa], 1)
            outputs, dec_states = dec_cell(current_input, dec_states)
            expanded_output = tf.concat([outputs, context, dec_inputa], axis=1)
            model = tf.layers.dense(expanded_output, n_class, activation=None)##단어 갯수만큼 출력을 맞춰줌
            probs = tf.nn.softmax(model)

            prediction = tf.argmax(model, 1)
            sente.append(model)
            predictions.append(prediction)

    predictions = tf.transpose(predictions)
    sente = tf.transpose(sente,(1,0,2))






cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=sente, labels=targets))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


sess = tf.Session()
sess.run(tf.global_variables_initializer())


def translate(imge):

    seq_data = ['','']#처음 데이터로 시작을 뜻하는 S만들어가면 되기때문에 둘다 비워둠

    output_batch, target_batch = make_batch([seq_data])##S만 들어가있는 데이터
    imge = [imge]

    # 결과가 [batch size, time step, input] 으로 나오기 때문에,
    # 2번째 차원인 input 차원을 argmax 로 취해 가장 확률이 높은 글자를 예측 값으로 만든다.

    print()


    result = sess.run(predictions,
                      feed_dict={images :imge,
                                 dec_input: output_batch})#여기서 S 다음에 올 글자 예측




###원래는 디코더 입력 데이터를 다 패딩처리해서 한꺼번에 넣어주었는데
###여기서는 다음단어를 예측하여 그다음 LSTM에 넣어줌

    for i in range(max):
        if result[0][i]==1:##끝을 뜻하는 E가 번호로 1로 표현되기 때문에 1이나오면 종료

            break
        elif i == 0:##result에 처음 데이터는 S다음 데이터이기 때문에 다음 output으로 S를 뜻하는 0과 re 를 같이 넣어줌
            re=result[0][i]
            output = [0,re]
            pad = [2] * (8)
            output = output + pad
            output_batch = []
            output_batch.append(np.eye(dic_len)[output])



            result = sess.run(predictions,
                              feed_dict={images :imge,
                                 dec_input: output_batch})




        else:##계속해서 result의 출력을 아웃풋에 추가시켜 단어를 예측
            re = result[0][i]
            output.append(re)
            pad = [2] * (max - (i + 2))
            output = output + pad
            output_batch = []
            output_batch.append(np.eye(dic_len)[output])
            result = sess.run(predictions,
                              feed_dict={images :imge,
                                 dec_input: output_batch})
            alq = sess.run(al,
                              feed_dict={images: imge,
                                         dec_input: output_batch})
        if i == max-1:##최대길이면 그냥 출력
            break


    # 결과 값인 숫자의 인덱스에 해당하는 글자를 가져와 글자 배열을 만든다.
    decoded = [char_arr[i] for i in result[0]]

    # 출력의 끝을 의미하는 'E' 이후의 글자들을 제거하고 문자열로 만든다.
    if 'E' in decoded:
        end = decoded.index('E')
        translated = ''.join(decoded[:end])
    else:##E가 없으면 모두출력
        translated = ''.join(decoded)

    return translated, alq

save_path = "./save/model.ckpt"
saver = tf.train.Saver()

saver.restore(sess,save_path=save_path)
c,v=translate(img2)

print(c)

import math

dx, dy = 0.001, 0.001
x = np.arange(-7, 7, dx)
y = np.arange(-7, 7, dy)

xmin, xmax, ymin, ymax = np.amin(x), np.amax(x), np.amin(y), np.amax(y)
extent = xmin, xmax, ymin, ymax
fig = plt.figure(frameon=False)
im1 = plt.imshow(v[2], cmap=plt.cm.gray, interpolation='nearest',
                             extent=extent)
print(v)
plt.show()