# OpenAI gym environment for QBDLM

<p style='text-align: justify;'>
In this document we present how to build the enviroment for the Bayesian Dynamic Linear Model (BDLM). The environemnt respects the requirement with respect to the <code>gym</code> environemnt, which is the standard environment for the OpenAI artificial intelligence research purposes.
</p>

## General framework

Based on openAI, the environment class has the following structure.

```python
import gym
from gym import error, spaces, utils
from gym.utils import seeding

class FooEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    ...
  def step(self, action):
    ...
  def reset(self):
    ...
  def render(self, mode='human'):
    ...
  def close(self):
    ...
```
In the simplest case, the class consists in an instructor and four methods to complete the agent-interaction loop as shown in the following figure.
<p align="center">
<img src="/assets/ai_loop.svg" alt="loop"
        title="agent-environemnt interaction" width="200"/>
</p>


<p style='text-align: justify;'>
Each time the agent take an <code>action</code> and the environment returns the <code>observation</code> and <code>reward</code>. The instructor <code>init()</code> assigns the properties for the environment instances, the method <code>step()</code> has an input <code>action</code> and it returns the <code>next state</code> (observation) of the agent and <code>reward</code>. Additionally, this method, has two other parameters namely <code>done</code> and <code>info</code>. <code>done</code> is a boolean indicating whether the interaction with the environemnt is done or not. In other words, it determines whetehr it is time for the <code>reset</code>. Finally, <code>info</code> reuturns addidtional information about the environemnt. Note that the agent does not use <code>info</code> during learning and the purpose of such return is for diagonstic information.
</p>
<p style='text-align: justify;'>
The <code>reset</code> method resets the episodes and return the initial state (observation) of the agent for the next episode. <code>render</code> and <code>close</code> are used for graphical purposes, which are not useful for our case (In future, maybe we utilize these features to graphically illustrate the agent environment interaction)
</p>
<p style='text-align: justify;'>
In the above figure (essentially for each environment), there are two spaces that need to be defined: action and observation spaces. To this end, gym provides two types of spaces, <code>discrete</code> and <code>box</code> spaces. The former allows a fixed non-negative range of numbers that is suitable for action space (e.g. 0 and 1 for two different actions). The latter represents an n-dimensional box, so a valid observation can be represented by this type of space. The box consists in a low and high parameters indicating the boundary of each dimension. For example, if the observation space is a 2-dimension space, then the parameters low and high are arrays of length 2. The reason for providing such generic format is to enable us to sample from the space or chech whether somethif belongs to it
</p>

__Discrete space example__
```python
import gym
from gym import spaces

space1 = spaces.Discrete(3) # A set with 3 elemenst {0, 1, 2}
print(space1)
x = space1.sample() # sampling from the space
print(x)
print(space1.contains(x)) # Check whether x belongs to the space
print(space1.n == 3) # Check the dimension of the space
```

__Box space example__
```python
import gym
from gym import spaces
import numpy as np

space2 = spaces.Box(low=np.array([-1.0, -2.0]),
        high=np.array([2.0, 4.0]), dtype=np.float32) # A 2 dimensional box space
print(space2.low, space2.high) # Print the boundaries of each dimension
x = space2.sample()
print(x)
print(space2.contains(x))
print(space2.shape==(2,))

```
## Environment for Anomaly Detection

The objective of the agent is to trigger an alarm in case of an anomaly. So, the action-space consists in two actions __Do Nothing__ or __Trigger an Alarm__. The environment that the agent interacts with is a set of hidden states obtained from Bayesian Dynamic Linear Modelling (BDLM) of the timeseries. For simplicity, we considrer two hidden states, namely __Baseline__ and __Trend__ that can carry anomalies. These two hidden states form a valued-vector representing the __state__ of the agent. We use the term __experience__ to indicate the set of states tha agent can has throught time. Therefore, at each time $t$ withing an experience, the agent experience a specific state corresponding to baseline and trend The enviroment can be discrete or continuous depending on the user choice. In general, the environment includes one constructor and eight methods following


- __Constructor__
  - <code>init</code>: Setting the enviroment properties
- __Methods__
  - <code>get_timeseries</code>: Loading the experience into the environment.
  - <code>set_alarmflag</code>: Indicating whether the agent has triggered the alarm.
  - <code>set_ref</code>: Setting the reference point (the state at time $t=0$).
  - <code>get_state_global</code>: Returning the absolute experience.
  - <code>get_state_local</code>: Returning the relative experience w.r.t the reference experience.
  - <code>seed</code>: Setting the see for reproductibility.
  - <code>step</code>: Returning the _next state_, _reward_, _done_, and _info_.
  - <code>reset</code>: Resetting the state of the agent at the beginning of each experience.
In what follows, constructor and methods are explained.

#### <code>init</code>
Initially, the constructor identifies the type of the space (discrete or continuous), so it creates proper space depending on the learning algorithm. Note that the nature of the space of the problem is continuous, but the discrete type is provided for validation purposes. Afterwards, the constructor sets the properties of the reward including the minimum reward $r_{\text{min}}$ <code>self.min_reward</code>, maximum reward $r_{\text{max}}$ <code>self.max_reward</code>, and false positive coefficient $\eta$ <code>self.etha</code>. In addition, the constructor set some properties related to the agent including the current timestamp index <code>self.timestamp</code>, a flag <code>self.alarmflag</code> to determine whether the agent has already triggered the alarm, and the reference state <code>self.reference</code> of the agent. <code>self.info</code> provides additional information about the environment. Note that thsi property is not used during the learning. <code>self.seed()</code> calles the <code>seed</code> method to set the seed number for reproductibility. Finally, <code>self.viewer</code> provides the properties for the graphical purposes. Here, we do not use this property.
```python
def __init__(self, stateType = 'Discrete'):

    if stateType == 'Discrete':
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(52)    

    elif stateType == 'Continuous':
        # states bounds
        self.min_level = -5
        self.max_level = 5
        self.min_trend = -0.005
        self.max_trend = 0.005

        self.low = np.array([self.min_level, self.min_trend])
        self.high = np.array([self.max_level, self.max_trend])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

    # reward properties
    self.min_reward = -1. # minimum reward
    self.max_reward = 1.  # maximum reward
    self.etha = 1. # false positive coefficient

    # agent properties
    self.timestamp = 0 # Current timestamps index
    self.alarmflag = 0 # Alarm trigerring flag
    self.reference = {} # Reference state

    self.info = []  # Additional information (it's not used for learning)

    self.seed() # Seed for reproductibility

    self.viewer = None # Viewer for graphical purposes (not used in this version)
```
#### <code>get_timeseries</code>
This method receives the experience in the form of a list contaning the hidden states, the list of timestamps in the form of date format, and a valued-vector representing the beginning and end of the anomaly window. The methods introduce three additional properties to the environment class. The characteristics of the input data will be discussed at the end of this document (Connecting openBDLM to QBDLM)
```python
def get_timeseries(self, bdlmData, timestampsValues, anomalyWindow):
    self.data = bdlmData
    self.anomalyWindow = anomalyWindow
    self.timestampsValues = timestampsValues
```

#### <code>set_alarmflag</code>
This method sets the alarm falg value. Two values can be assigned, 0 or 1, respectively indicating that the agent has _not triggered the alarm_ and _triggered the alarm_. The default value is 0. 
```python
def set_alarmflag(self, flagValue=0):
    self.alarmflag = flagValue
```
#### <code>set_ref</code>
This method sets the reference state $S_{\text{reference}}$. It receives the state as numpy array and assigns it to the reference attribute.

```python
def set_ref(self, state):
    self.reference = state
```
#### <code>get_state_global</code>
This method returns the absolute state values of the agent based on the timestamp index as the input. 

```python
def get_state_global(self, t):
    return np.array([self.data['Baseline'][t], self.data['Trend'][t]])
```
#### <code>get_state_local</code>
This method returns the relative state values of the agent based on the timestamp index as the input. Note that the relationship between the absolute state $S_{\text{absolute}}(t)$ and relative state $S_{\text{relative}}(t)$ is defined as $S_{\text{relative}}(t)=S_{\text{absolute}}(t)-S_{\text{reference}}$

```python
def get_state_local(self, t):
    return self.get_state_global(t) - np.array([self.reference[0],0.0])
```

#### <code>seed</code>
This method sets the seed number for reproductibility of the results.
```python
def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]
```

#### <code>step</code>
This method receives the action as the input and returns the <code>next state</code>, <code>reward</code>, <code>done</code>, and <code>info</code>. The next state is the state corresponding to the next timestamps in the experience. The reward is calculated based on the confusiuon matrix. The boolean _done_ is determined based on the index of the timestamps in the experience, and info provides additional information regarding the agent-environment interaction.
```python
def step(self, action):
  
    t = self.timestamp
    currentTime = self.timestampsValues['timestamps'][t]
    self.timestamp = t+1
    
    done = bool(t >= len(self.data['Baseline'])-1)

    if self.anomalyWindow.all:
        if currentTime < self.anomalyWindow[0] or currentTime > self.anomalyWindow[1]:
            if action == 0:
                reward = self.max_reward  # True Positive
            else:
                reward = self.min_reward  # False Negative (False Alarm)
        elif currentTime >= self.anomalyWindow[0] and currentTime <= self.anomalyWindow[1]:
            if action == 0:
                # False Positive (Delayed Alarm)
                if self.alarmflag == 0:
                    reward = - self.etha * \
                        np.abs(self.reference[0] -
                               self.data['Baseline'][t])
                else:
                    reward = self.max_reward
            else:
                if self.alarmflag == 0:
                    # True Negative for first time triggering.
                    reward = self.max_reward
                    self.alarmflag = 1
                else:
                    # False Negative (triggers second time).
                    reward = self.min_reward
    else:
        if action == 0:
            reward = self.max_reward
        else:
            reward = self.min_reward

    if action == 1:
        self.reference = self.get_state_global(t+1)
    
    
    self.state = self.get_state_local(t+1)

    self.info.append([t, t+1])

    return np.array(self.state), reward, done, np.array(self.info)
```
#### <code>reset</code>
This method resets the reference, state, timestamps, and info at the begining of each experience. The reset is performed based on the user's preference regarding resetting with or without considering the reference point.
```python
def reset(self, withReference = True):
         
    if withReference:
        self.reference = self.get_state_global(0)
        self.state = self.get_state_local(0)
    else:
        self.reference = self.get_state_global(0)
        self.state = self.get_state_global(0)
    
    self.timestamp = 0
    self.info = []
    return np.array(self.state)
```
Two remaining concerns are the neccessary packages and the types of the data as the experience to be passed to the environment.

## Requiered Packages
The following packages are requiered to build the environemnt.

```python
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
```
## Import data (connecting BDLM to QBDLM)
The output of the BDLM is a mat file that need to be processed and structured to be able to be utilized for QBDLM. This is performed by using two functions, <code>datenum_to_datetime</code> and <code>createData</code>. the former is used to convert the MATLAB timestamps to Python timestams, and the latter is utilized to extract and construct the experience for the QBDLM

```python
def datenum_to_datetime(datenum):
    """
    Convert Matlab datenum into Python datetime.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    days = datenum % 1
    return datetime.fromordinal(int(datenum)) \
        + timedelta(days=days) \
        - timedelta(days=366)


def createData(filename):
    mat = scipy.io.loadmat(filename)
    values = mat['values']
    values[:, 0] = values[:, 0]/1000
    timestamps = mat['timestamps']
    if mat['flag'].item()==1:
        anomalyWindow = mat['window'][0]
    else:
        anomalyWindow = {}


    baseLine = pd.Series(values[:, 0], index=timestamps)
    trend = pd.Series(values[:, 1], index=timestamps)
    acc = pd.Series(values[:, 2], index=timestamps)

    timesmatlab = timestamps[:, 0].tolist()

    pydate = []
    for i in timesmatlab:
        pydate.append(datenum_to_datetime(i))

    newTimestamps = [pydate[x].strftime('%Y-%m-%d')
                     for x in range(len(pydate))]

    matDic = {'timestamps': newTimestamps,
              'Baseline': values[:, 0].tolist(),
              'Trend': values[:, 1].tolist(),
              'Acceleration': values[:, 2].tolist()}
    
    timeDic = {'timestamps': timesmatlab}

    matFrame2 = pd.DataFrame(
        matDic,  columns=['timestamps', 'Baseline', 'Trend', 'Acceleration'])
    matFrame2.set_index("timestamps", drop=True, inplace=True)

    timestampsValues = pd.DataFrame(timeDic,  columns=['timestamps'])
    return (matFrame2, timestampsValues, anomalyWindow)
``` 




