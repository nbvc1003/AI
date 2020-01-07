import numpy as np
import matplotlib.pyplot as plt
image = np.array([[
                [[1],[2],[3]],
                [[4],[5],[6]],
                [[7],[8],[9]]
                    ]], dtype=np.float32)
print(image.shape)
print(image.reshape(3,3))
# plt.imshow(image.reshape(3,3)) # 숫자가 작을수록 어두움
plt.imshow(image.reshape(3,3), cmap='Greys') # 숫자만큼 회색 반영..
plt.show()