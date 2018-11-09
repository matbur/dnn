# dnn

## Usage

```bash
pip install sdnn
```

```python
import numpy as np

import sdnn

x = np.array([ [1, 1], [1, 0], [0, 1], [0, 0], ], dtype=float)
y = np.array([ [1, 0], [0, 1], [0, 1], [1, 0], ], dtype=float)

net = sdnn.input_data((None, 2))
net = sdnn.fully_connected(net, 3, activation='tanh')
net = sdnn.fully_connected(net, 2, activation='tanh')

model = sdnn.Model(net)
model.fit(x, y, n_epoch=200)
model.save('xor_model.json')
model.load('xor_model.json')

for i in zip(y, model.predict(x)):
    print(*i)

model.plot_error()
```
