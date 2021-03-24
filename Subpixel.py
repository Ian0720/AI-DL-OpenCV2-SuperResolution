from keras import backend as K
from keras.layers import Conv2D

"""
	Conv2D의 하위 클래스로서의 하위 픽셀 계층.

    이 계층은 dilation_rate()를 제외한 모든 정규 인수를 허용합니다. 인수 r은 Conv2D의 정상 출력에 적용되는 업샘플링 계수를 나타냅니다.

    이 계층의 출력은 표시된 필터 필드와 동일한 수의 채널을 가지므로 그레이스케일, 색상 또는 숨겨진 레이어로 작동합니다.

	Arguments:
        *Distilation_rate()가 제거되었다는 점에 유의하여 Conv2Dags용 Keras Docs 참조*
        r: 정상 Conv2DA 테스트의 출력에 적용되는 업스케일링 팩터가 포함되어 Cifar10 데이터 세트에서 초해상도 수행합니다.

        이러한 영상은 작으므로 스케일 팩터 2만 사용됩니다. 테스트 영상은 'test_output/' 디렉토리에 저장됩니다. 
        이 테스트는 5개의 epoch 동안 실행되며, 132행에서 변경할 수 있습니다. 다음 명령을 사용하여 이 테스트를 실행할 수 있습니다.

        mkdir test_output
        python keras_subpixel.py
"""


class Subpixel(Conv2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 r,
                 padding='valid',
                 data_format=None,
                 strides=(1,1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Subpixel, self).__init__(
            filters=r*r*filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.r = r

    def _phase_shift(self, I):
        r = self.r
        bsize, a, b, c = I.get_shape().as_list()
        bsize = K.shape(I)[0] # 정의되지 않은 디멘션 논 타입에 대하여 핸들링해주는 부분입니다.
        X = K.reshape(I, [bsize, a, b, int(c/(r*r)),r, r]) # bsize, a, b, c/(r*r), r, r
        X = K.permute_dimensions(X, (0, 1, 2, 5, 4, 3))  # bsize, a, b, r, r, c/(r*r)
                              # Keras 백엔드는 tf.split을 지원하지 않습니다, 해서 향후 미래에 나올 버전에서는 이것이 더 나을수도 있어요!
        X = [X[:,i,:,:,:,:] for i in range(a)] # a, [bsize, b, r, r, c/(r*r)
        X = K.concatenate(X, 2)  # bsize, b, a*r, r, c/(r*r)
        X = [X[:,i,:,:,:] for i in range(b)] # b, [bsize, r, r, c/(r*r)
        X = K.concatenate(X, 2)  # bsize, a*r, b*r, c/(r*r)
        return X

    def call(self, inputs):
        return self._phase_shift(super(Subpixel, self).call(inputs))

    def compute_output_shape(self, input_shape):
        unshifted = super(Subpixel, self).compute_output_shape(input_shape)
        return (unshifted[0], self.r*unshifted[1], self.r*unshifted[2], int(unshifted[3]/(self.r*self.r)))

    def get_config(self):
        config = super(Conv2D, self).get_config()
        config.pop('rank')
        config.pop('dilation_rate')
        config['filters'] = int(config['filters'] / self.r*self.r)
        config['r'] = self.r
        return config
