import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy import stats
import keras
from keras.utils import to_categorical
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt

column_names = ['ids','temp','hum','co2','num_people']
n = 6
outs = np.arange(2*n).reshape(-1,1)

df1 = pd.read_csv('data18.csv', header=None,names=column_names)
df1 = df1.apply(pd.to_numeric)
df1.temp = savgol_filter(df1.temp, 101, 3)
df1.hum = savgol_filter(df1.hum, 101, 3)
df1.co2 = savgol_filter(df1.co2, 101, 3)

df2 = pd.read_csv('data19.csv', header=None,names=column_names)
df2 = df2.apply(pd.to_numeric)
df2.temp = savgol_filter(df2.temp, 101, 3)
df2.hum = savgol_filter(df2.hum, 101, 3)
df2.co2 = savgol_filter(df2.co2, 101, 3)

df3 = pd.read_csv('data20.csv', header=None,names=column_names)
df3 = df3.apply(pd.to_numeric)
df3.temp = savgol_filter(df3.temp, 101, 3)
df3.hum = savgol_filter(df3.hum, 101, 3)
df3.co2 = savgol_filter(df3.co2, 101, 3)

df1_train = df1[df1['ids']<=16000]
df1_test = df1[df1['ids']>16000]

df2_train = df2[df2['ids']<=16000]
df2_test = df2[df2['ids']>16000]

df3_train = df3[df3['ids']<=16000]
df3_test = df3[df3['ids']>16000]

result_frames = [df1, df2, df3]
result = pd.concat(result_frames)

#result.to_excel("op.xlsx")

test_frames = [df1_test, df2_test, df3_test]
train_frames = [df1_train, df2_train, df3_train]
final_test = pd.concat(test_frames)
final_train = pd.concat(train_frames)

def create_dataset(X, y, time_steps=1, step=1):
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i:(i + time_steps)].values
        labels = y.iloc[i: i + time_steps]
        Xs.append(v)
        ys.append(stats.mode(labels)[0][0])
    return np.array(Xs), np.array(ys).reshape(-1, 1)

TIME_STEPS = 300
STEP = 300

X_train, y_train = create_dataset(
    final_train[['temp', 'hum', 'co2']],
    final_train.num_people,
    TIME_STEPS,
    STEP
)
y2_train = np.zeros((y_train.shape[0],y_train.shape[1]))


for i in range(1,len(y_train)):
    y2_train[i] = y_train[i] - y_train[i-1] + n
y2_train[0] = y2_train[1]

X_test, y_test = create_dataset(
    final_test[['temp', 'hum', 'co2']],
    final_test.num_people,
    TIME_STEPS,
    STEP
)

y2_test = np.zeros((y_test.shape[0],y_test.shape[1]))

for i in range(1,len(y_test)):
    y2_test[i] = y_test[i] - y_test[i-1] + n

y2_test[0] = y2_test[1];

#print("X_test", X_test.shape)
#print("y_test", y2_test.shape)
#print("X_train", X_train.shape)
#print("y_train", y2_train.shape)

enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

enc = enc.fit(outs)

y2_train = enc.transform(y2_train)
y2_test = enc.transform(y2_test)

model = keras.Sequential()
model.add(
    keras.layers.Bidirectional(
      keras.layers.LSTM(
          units=16,
          input_shape=[X_train.shape[1], X_train.shape[2]]
      )
    )
)
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(units=64, activation='relu'))
model.add(keras.layers.Dense(2*n, activation='softmax'))

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['acc']
)

history = model.fit(
    X_train, y2_train,
    epochs=20,
    batch_size=32,
    validation_split=0.25,
    shuffle=False
)


'''
while(121):
    #for the next 5 minutes:
        #read co2 data and append to array
        #read temp data and append to array
        #read hum data and append to array
    #Join the 3 arrays as shown in the report to make 3D input matrix of size (1,300,3)
    #y = model.predict(input) --> this gives a softmax output of 1 row and 2*n columns.
    #y2 = index of max value in y
    #y3 = y2 - 6 --> which gives how many people have been added or subtracted in the room for that signature
    #print(timestamp, y3)
    
'''    
        
    
    


