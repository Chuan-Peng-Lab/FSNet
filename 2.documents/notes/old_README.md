# PsyNN
*详细文档请见19200210+孙禾嘉+开题报告_rev_hcp.doc*

0-datasets：使用的所有数据内容，详见readme

1-documents：部分文献和笔记，作图文件

2-rnn_code：存放代码

3-tools：相关的包文件

4-misc：杂项

———————————————————————

## 目前进度总结

熟悉了使用PsychRNN包进行Task定义，网络训练，网络应用的全流程
 尝试了基于Gym的Neuro包
 了解了一些文章中使用神经网络进行研究分析的方法（需要具体整理）

下阶段To do：
 如何根据网络输出的权重分析网络结构
 如何对网络训练施加生物学限制



## 研究目的

使用神经网络，通过对被试在行为实验中的数据的训练，获得模拟被试行为的网络。可以在此网络的基础上，研究具体行为操作中的关联和特点。



## 研究工具

目前现有的训练被试的神经网络包有NeuroGym包和PsychRNN包

PsychRNN基于python和TensorFlow，使用循环神经网络（RNN）作为训练网络，采用前后端分离的设计，使心理学研究者能够方便的自定义实验任务而无需关心内部训练过程。

NeuroGym包基于OpenAI的工具包Gym，提供了多种训练方法和实验范式。

PsychRNN为心理学研究优化，提供了便捷的自定义Task的方法，并且没有太多复杂的设置，适合对于机器学习相关领域并不熟悉的研究者使用。而NeuroGym基于广泛使用的Gym包，适用性强，可扩展丰富，自定义内容多，但是使用没有前者便利（需要掌握Gym包的使用）。

### PsychRNN：

RNN适用于处理时间序列编码的数据，而认知实验的数据结果可以采用时间编码，适合让RNN来学习。

PsychRNN将会把整个实验过程通过N_in输入通道和N_out输出通道的数值编码成Ndarry序列，RNN提供了相应的文档说明来帮助建立自己的Task，PsychRNN将会通过Task设定的规则编写实验序列。

在训练部分，可以对神经网络进行参数调整，以及施加生物学限制，更好的模拟人类被试的反应情况。

使用PsychRNN的流程大致如下

•       定义Task -> 

•       由Task生成编码后的实验和反应序列 -> 

•       用序列训练RNN神经网络 -> 

•       使用神经网络作为被试进行实验 -> 

•       针对权重进行操作或分析，研究实验的内部过程

#### 建立自定义Task——代码分析

要在PsychRNN中自定义Task，需要自行编写类，继承自Task父类。Task继承自ABC抽象基类，设定了一些Task的预先方法。

from abc import ABCMeta, abstractmethod
 \# abstract class python 2 & 3 compatible
 ABC = ABCMeta('ABC', (object,), {})
 \#继承自抽象基类
 class Task(ABC):

from psychrnn.tasks.task import Task
 import numpy as np
 \#自定义Task继承自Task父类
 class SimplePD(Task):

##### **init**初始化函数

\#初始化函数,包含任务参数设定和计算
 def __init__(self, N_in, N_out, dt, tau, T, N_batch):
   \# ----------------------------------
   \# 初始化需要的参数
   \# ----------------------------------
   self.N_batch = N_batch
   self.N_in = N_in
   self.N_out = N_out
   self.dt = dt
   self.tau = tau
   self.T = T

   \# ----------------------------------
   \# Calculate implied parameters
   \# ----------------------------------
   self.alpha = (1.0 * self.dt) / self.tau
   self.N_steps = int(np.ceil(self.T / self.dt))

   \# Initialize the generator used by get_trial_batch
   \# 初始化一个生成器，用来生成序列
   self._batch_generator = self.batch_generator()



##### trial_funtion函数

这是定义实验操作的关键函数。PsychRNN将实验过程表示为以dt为最小单位，长度为T的时间序列，每一个dt时间片段被单独处理，trial_funtion就是处理函数。整个过程会对trial_funtion进行不断循环。

要自定义实验task，主要编写这个函数

def trial_function(self, time, params): 
   """ Compute the trial properties at the given time. 
   Based on the params compute the trial stimulus (x_t), correct output (y_t), and mask (mask_t) at the given time. 
   Args: 
     time (int): The time within the trial (0 <= time < T). 
     params (dict): The trial params produced generate_trial_params() 
   Returns: 
     tuple: 
     x_t (ndarray(dtype=float, shape=(N_in,))): Trial input at time given params. 
     y_t (ndarray(dtype=float, shape=(N_out,))): Correct trial output at time given params. 
     mask_t (ndarray(dtype=bool, shape=(N_out,))): True if the network should train to match the y_t, False if the network should ignore y_t when training. 

   """ 
   stim_noise = 0.1 
   \# 在T/4时间点开始 
   onset = self.T/4.0 
   \# 刺激长度为T/2 
   stim_dur = self.T/2.0 
   \# ---------------------------------- 
   \# Initialize with noise 
   \# ---------------------------------- 
   \# 输入通道为噪声 
   x_t = np.sqrt(2*self.alpha*stim_noise*stim_noise)*np.random.randn(self.N_in) 
   \# 输出通道为空 
   y_t = np.zeros(self.N_out) 
   \# 遮罩为空，这个变量用于确定当前输出值是否有意义（是否是被试反应阶段） 
   mask_t = np.ones(self.N_out) 

   \# ---------------------------------- 
   \# Retrieve parameters 
   \# ---------------------------------- 
   coh = params['coherence'] 
   direction = params['direction'] 

   \# ---------------------------------- 
   \# Compute values 
   \# 这是每个时间片段的处理过程，也是实验序列的过程，此代码表示的是perceptual discrimination的task过程。 
   \# ---------------------------------- 
   if onset < time < onset + stim_dur: 
     x_t[direction] += 1 + coh 
     x_t[(direction + 1) % 2] += 1 

   if time > onset + stim_dur + 20: 
     y_t[direction] = 1. 

   if time < onset + stim_dur: 
     mask_t = np.zeros(self.N_out) 

   return x_t, y_t, mask_t

通过对Task规则的定义，通过batch_generator生成器，PsychRNN能够生成用于训练的相应试验任务的数据序列。

\#---------------------- 设置Task并训练神经网络 ---------------
 pd = SimplePD(dt = 10, tau = 100, T = 2000, N_batch = 128)
 network_params = pd.get_task_params() # get the params passed in and defined in pd
 network_params['name'] = 'model' # name the model uniquely if running mult models in unison
 network_params['N_rec'] = 50 # set the number of recurrent units in the model
 model = Basic(network_params) # instantiate a basic vanilla RNN

 model.train(pd) # train model to perform pd task

x,target_output,mask, trial_params = pd.get_trial_batch() 
 \# 根据Task条件生成一些输入刺激，并输入到神经网络模型中 
 model_output, model_state = model.test(x) 

 plt.plot(model_output[0,:,:]) 

![标题: fig:](README/clip_image002.png)

结果如图，曲线绘制的是神经网络在两个输出通道的反应结果，符合实验Task对于被试反应输出的期待。



### NeuroGym

（因为涉及的内容较多，对这个包还没有进行深入的研究）

如下是在NeuroGym中建立Task和训练网格的代码过程

\# Task name 
 name = 'PerceptualDecisionMaking-v0' 
 \# task specification (here we only specify the duration of the different trial periods) 
 timing = {'fixation': ('constant', 300), 
      'stimulus': ('constant', 500), 
      'decision': ('constant', 300)} 
 kwargs = {'dt': 100, 'timing': timing} 
 \# build task 
 \# 通过参数和预先设定好的PerceptualDecisionMaking任务范式构建实验程序 
 env = gym.make(name, **kwargs) 
 \# wrapp task with pass-reward wrapper 
 env = pass_reward.PassReward(env) 
 \#对神经网络进行训练 
 env = DummyVecEnv([lambda: env]) 
 model = A2C(LstmPolicy, env, verbose=1, policy_kwargs={'feature_extraction':"mlp"}) 
 model.learn(total_timesteps=100000, log_interval=1000) 
 env.close() 

[neurogym/example_neurogym_rl.ipynb at master · neurogym/neurogym · GitHub](https://github.com/neurogym/neurogym)
 [Welcome to PsychRNN’s documentation! — PsychRNN 1.0.0 documentation](https://psychrnn.readthedocs.io/en/latest/index.html)



 
