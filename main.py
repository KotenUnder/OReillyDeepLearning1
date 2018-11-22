import numpy as np
import matplotlib.pylab as plt
import NeuralNetwork as nn

x = np.arange(-5.0, 5.0, 0.1)
y = nn.step_function_array(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()



