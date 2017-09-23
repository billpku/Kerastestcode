# visualize the training process with plot

from keras.models import Sequential
from keras.layers import  Dense
import matplotlib.pyplot as plt
import  numpy

#fix the random number
seed = 8
numpy.random.seed(seed)

#load the data
data = numpy.loadtxt('pima-indians-diabetes.csv',delimiter=',')
X = data[:,0:8]
Y = data[:,8]

# define the sequential model
model = Sequential()
model.add(Dense(8,input_dim=8,init='uniform',activation='relu'))
model.add(Dense(12,init='uniform',activation='relu'))
model.add(Dense(1,init='uniform',activation='sigmoid'))

# compile the model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# fit the model and give the info to variable history
history = model.fit(X,Y,validation_split=0.33,nb_epoch=200,batch_size=15,verbose=0)

# print the history
print (history.history.keys())

# set a plot for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
#title
plt.title('model accuracy')
# x dimension label
plt.xlabel('epoch')
# y dimension label
plt.ylabel('accuracy')
# plot info setting
plt.legend(['train','test'],loc='upper left')
# show the plot
plt.show()


# set a plot for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
#title
plt.title('model loss')
# x dimension label
plt.xlabel('epoch')
# y dimension label
plt.ylabel('loss')
# plot info setting
plt.legend(['train','test'],loc='upper left')
# show the plot
plt.show()