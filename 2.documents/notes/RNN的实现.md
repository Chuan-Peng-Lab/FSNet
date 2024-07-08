# RNN的实现



## 对ht的更新过程

$$
h_t = f_W(h_(t-1),x_t)
$$

$$
h_t = tanh(W_(hh)*h_(t-1),W_(xh)*x_t)
$$

ht：记忆

W_hh：对ht做特征提取
W_xh：对xt做特征提取

tanh：激活函数，在RNN里常用

y_t = W_hy*h_t，得到一个输出经过线性层

## 对参数的更新过程

![rnn1](RNN的实现/rnn1.png)

W_R:Whh
W_I:Whx
通过以下方法计算：

![image-20230826194838786](RNN的实现/image-20230826194838786-16930505239932.png)

从0时刻到t时刻的累加
