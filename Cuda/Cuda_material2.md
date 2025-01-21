# Cuda并行编程

### CPU和GPU

- **主机：**将CPU及系统的内存(内存条)称为主机;
- **设备：**将GPU及GPU本身的显示内存称为设备;
- **线程(Thread)：**一般通过GPU的一个核进行处理;
- **[线程块](https://zhida.zhihu.com/search?content_id=209214184&content_type=Article&match_order=1&q=线程块&zhida_source=entity)(Block)：**由多个线程组成；各block是并行执行的，block间无法通信，也没有执行顺序。
- **[线程格](https://zhida.zhihu.com/search?content_id=209214184&content_type=Article&match_order=1&q=线程格&zhida_source=entity)(Grid)：**由多个线程块组成。
- **[核函数](https://zhida.zhihu.com/search?content_id=209214184&content_type=Article&match_order=1&q=核函数&zhida_source=entity)(Kernel)：**在GPU上执行的函数通常称为核函数;一般通过标识符__global__修饰，调用通过<<<参数1,参数2>>>，用于说明内核函数中的线程数量，以及线程是如何组织的。

GPU与CPU通过[PCIe总线](https://zhida.zhihu.com/search?content_id=6024941&content_type=Article&match_order=1&q=PCIe总线&zhida_source=entity)连接在一起来协同工作，CPU所在位置称为为主机端（host），而GPU所在位置称为设备端（device）

总线：

PCI, 全称Peripheral Component Interconnect, 是一种连接处理器和外部设备的总线标准 (其中外围设备包括计算机显卡, 声卡, 硬盘驱动适配器，SSDs, WIFI和有线网卡等等)。

PCIe，全称Peripheral Component Interconnect Express，是一种作为PCI总线替代品的高速串行通信标准，是一种全双工通道。

### 基本cuda知识

![image-20250121173445472](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20250121173445472.png)

可以看到GPU包括更多的运算核心，其特别适合数据并行的计算密集型任务，如大型矩阵运算，而CPU的运算核心较少，但是其可以实现复杂的逻辑运算，因此其适合控制密集型任务。另外，CPU上的线程是重量级的，上下文切换开销大，但是GPU由于存在很多核心，其线程是轻量级的。因此，基于CPU+GPU的异构计算平台可以优势互补，CPU负责处理逻辑复杂的串行程序，而GPU重点处理数据密集型的并行计算程序，从而发挥最大功效。

cuda是NVIDIA公司开发的GPU编程模型，它提供了对GPU编程的简易接口，基于cuda编程可以实现GPU的简易程序

查看显卡支持的cuda版本：nvidia-smi	查看cuda版本：nvcc -V

典型的CUDA程序的执行流程如下：

1. 分配host内存，并进行数据初始化；
2. 分配device内存，并从host将数据拷贝到device上；
3. 调用CUDA的[核函数](https://zhida.zhihu.com/search?content_id=6024941&content_type=Article&match_order=1&q=核函数&zhida_source=entity)在device上完成指定的运算；
4. 将device上的运算结果拷贝到host上；
5. 释放device和host上分配的内存。

kernel是在device上线程中并行执行的函数，核函数用`__global__`符号声明，在调用时需要用`<<<grid, block>>>`来指定kernel要执行的线程数量，**在CUDA中，每一个线程都要执行核函数，并且每个线程会分配一个唯一的线程号thread ID，这个ID值可以通过核函数的内置变量`threadIdx`来获得**。

由于GPU实际上是异构模型，所以需要区分host和device上的代码，在CUDA中是通过函数类型限定词开区别host和device上的函数，主要的三个函数类型限定词如下：

- `__global__`：在device上执行，从host中调用（一些特定的GPU也可以从device上调用），返回类型必须是`void`，不支持可变参数参数，不能成为类成员函数。注意用`__global__`定义的**kernel是异步的，这意味着host不会等待kernel执行完就执行下一步。**
- `__device__`：在device上执行，**单仅可以从device中调用**，不可以和`__global__`同时用。
- `__host__`：在host上执行，仅可以从host上调用，**一般省略不写，不可以和`__global__`同时用，但可和`__device__`，此时函数会在device和host都编译。**

首先GPU上很多并行化的轻量级线程。kernel在device上执行时实际上是启动很多线程，**一个kernel所启动的所有线程称为一个网格（grid）**，同一个网格上的线程共享相同的全局内存空间，grid是线程结构的第一层次，而网格又可以分为很多**线程块**（block），一个线程块里面包含很多线程，这是第二个层次。线程两层组织结构如下图所示，这是一个gird和block均为2-dim的线程组织。grid和block都是定义为`dim3`类型的变量，`dim3`可以看成是包含三个无符号整数（x，y，z）成员的结构体变量，在定义时，缺省值初始化为1。因此grid和block可以灵活地定义为1-dim，2-dim以及3-dim结构，对于图中结构（主要水平方向为x轴），定义的grid和block如下所示，kernel在调用时也必须通过[执行配置](https://link.zhihu.com/?target=http%3A//docs.nvidia.com/cuda/cuda-c-programming-guide/index.html%23execution-configuration)`<<<grid, block>>>`来指定kernel所使用的线程数及结构。

```c
dim3 grid(3, 2);
dim3 block(5, 3);
kernel_fun<<< grid, block >>>(prams...);
```

![image-20250121174603287](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20250121174603287.png)

所以，一个线程需要两个内置的坐标变量（blockIdx，threadIdx）来唯一标识，它们都是`dim3`类型变量，其中blockIdx指明线程所在grid中的位置，而threaIdx指明线程所在block中的位置，如图中的Thread (1,1)满足：

```text
threadIdx.x = 1
threadIdx.y = 1
blockIdx.x = 1
blockIdx.y = 1
```



一个线程块上的线程是放在同一个流式多处理器（SM)上的，但是单个SM的资源有限，这导致线程块中的线程数是有限制的，现代GPUs的线程块可支持的线程数可达1024个。

有时候，我们要知道一个线程在blcok中的全局ID，此时就必须还要知道block的组织结构，这是通过线程的内置变量blockDim来获得。它获取线程块各个维度的大小。对于一个2-dim的block (Dx,Dy) ，线程 (x,y) 的ID值为 (x+y∗Dx) ，如果是3-dim的block (Dx,Dy,Dz) ，线程 (x,y,z) 的ID值为 (x+y∗Dx+z∗Dx∗Dy) 。另外线程还有内置变量gridDim，用于获得网格块各个维度的大小。

```
// Kernel定义
// 两个矩阵相加的cuda函数写法

__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]) 
{ 
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y; 
    if (i < N && j < N) 
        C[i][j] = A[i][j] + B[i][j]; 
}
int main() 
{ 
    ...
    // Kernel 线程配置
    dim3 threadsPerBlock(16, 16); 
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    // kernel调用
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C); 
    ...
}
```

此外这里简单介绍一下CUDA的内存模型，如下图所示。可以看到，每个线程有自己的私有本地内存（Local Memory），而每个线程块有包含共享内存（Shared Memory）,可以被线程块中所有线程共享，其生命周期与线程块一致。此外，所有的线程都可以访问全局内存（Global Memory）。还可以访问一些只读内存块：常量内存（Constant Memory）和纹理内存（Texture Memory）。内存结构涉及到程序优化，这里不深入探讨它们。

![image-20250121180853917](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20250121180853917.png)

一个kernel实际上会启动很多线程，这些线程是逻辑上并行的，但是在物理层却并不一定。这其实和CPU的多线程有类似之处，多线程如果没有多核支持，在物理层也是无法实现并行的。但是好在GPU存在很多CUDA核心，充分利用CUDA核心可以充分发挥GPU的并行计算能力

GPU硬件的一个核心组件是SM，前面已经说过，SM是英文名是 **Streaming Multiprocessor**，翻译过来就是**流式多处理器**。

SM的核心组件包括CUDA核心，共享内存，寄存器等，SM可以并发地执行数百个线程，并发能力就取决于SM所拥有的资源数。当一个kernel被执行时，它的gird中的线程块被分配到SM上，一个线程块只能在一个SM上被调度。SM一般可以调度多个线程块，这要看SM本身的能力，那么有可能一个kernel的各个线程块被分配多个SM，**所以grid只是逻辑层，而SM才是执行的物理层。**

SM采用的是[SIMT](https://link.zhihu.com/?target=http%3A//docs.nvidia.com/cuda/cuda-c-programming-guide/index.html%23simt-architecture) (Single-Instruction, Multiple-Thread，单指令多线程)架构，基本的执行单元是**线程束（warps)**，线程束包含32个线程，这些线程同时执行相同的指令，但是每个线程都包含自己的指令地址计数器和寄存器状态，也有自己独立的执行路径。

所以尽管线程束中的线程同时从同一程序地址执行，但是可能具有不同的行为，比如遇到了分支结构，一些线程可能进入这个分支，但是另外一些有可能不执行，它们只能死等，因为GPU规定线程束中所有线程在同一周期执行相同的指令，线程束分化会导致性能下降。

**当线程块被划分到某个SM上时，它将进一步划分为多个线程束，因为这才是SM的基本执行单元**，但是一个SM同时并发的线程束数是有限的。这是因为资源限制，SM要为每个线程块分配共享内存，而也要为每个线程束中的线程分配独立的寄存器。以SM的配置会影响其所支持的线程块和线程束并发数量。总之，就是网格和线程块只是逻辑划分，一个kernel的所有线程其实在物理层是不一定同时并发的。所以kernel的grid和block的配置不同，性能会出现差异，这点是要特别注意的。还有，由于SM的基本执行单元是包含32个线程的线程束，所以block大小一般要设置为32的倍数。

![image-20250121193642157](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20250121193642157.png)