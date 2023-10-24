import tensorflow as tf
class MyLoss(tf.keras.losses.Loss):
    
    def call(self, y_true, y_pred):
        y_0_true = y_true[:,0]
        y_1_true = y_true[:,1]
        y_0 = y_pred[:,0]
        y_1 = y_pred[:,1]
        y_2 = y_pred[:,2]
        y_3 = y_pred[:,3]
        y_4 = y_pred[:,4]
        first  = 2*(y_2+y_3)
        second = tf.square(y_0-y_0_true)*(tf.exp(-2*y_2))
        third  = tf.square(y_1-y_1_true)*(tf.exp(-2*y_3))
        forth = -2*y_4*(y_0-y_0_true)*(y_1-y_1_true)*(tf.exp(-2*y_3-y_2))
        fifth = (tf.square(y_4))*(tf.square(y_0-y_0_true))*(tf.exp(-2*y_2-2*y_3))
        loss = first+second+third+forth+fifth
        return tf.reduce_mean(loss)
    
    def log_det_Sigma(self, y_true, y_pred):
        y_2 = y_pred[:,2]
        y_3 = y_pred[:,3]
        return 2*(y_2+y_3)
    
    def chisq(self, y_true, y_pred):
        return self.call(y_true, y_pred)-self.log_det_Sigma(y_true, y_pred)
    
    def MSE(self, y_true, y_pred):
        y_0_true = y_true[:,0]
        y_1_true = y_true[:,1]
        y_0 = y_pred[:,0]
        y_1 = y_pred[:,1]
        squared_difference = (tf.square(y_0_true - y_0)+tf.square(y_1_true - y_1))
        return tf.reduce_mean(squared_difference)