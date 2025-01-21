# Cuda并行编程

### 科学计算

科学计算的三个步骤：读取数据 -> 计算 -> 写回数据

计算所需要的时间明显小于读取数据和写回数据所需要的时间

对应的冯诺依曼计算机体系架构：

![image-20250121104617661](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20250121104617661.png)

![image-20250121105704246](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20250121105704246.png)



### 并行计算三大定律

#### Amdahl's Law 阿姆达尔定律

适用情景：总工作量固定

![image-20250121110030287](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20250121110030287.png)

#### Gustafson's Law 古斯塔法森定律

思路：总工作量固定的情况下，processor数量上升，但加速比存在上限

![image-20250121110804048](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20250121110804048.png)

保证工作量随着处理器的缩放，保持处理时间固定

#### Sun-Ni's Law 孙-倪定律

![image-20250121111108551](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20250121111108551.png)

G(N)不一定是线性函数，可以是非线性函数

### Roof Line Model 与深度学习性能分析

在真实世界中，任何模型（例如 VGG / MobileNet 等）都必须依赖于具体的计算平台（例如CPU / GPU / ASIC 等）才能展现自己的实力。此时，模型和计算平台的"默契程度"会决定模型的实际表现。**Roofline Model 提出了使用 Operational Intensity（计算强度）进行定量分析的方法，并给出了模型在计算平台上所能达到理论计算性能上限公式**

