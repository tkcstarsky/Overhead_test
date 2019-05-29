# Overhead测试
本内容实现了对稀疏矩阵与密矩阵、for loop的两个Overhead的测试
******
## 目录文件说明
* src/decoding_test.cu 密矩阵的csr decoding Overhead测试
* src/loop_test.cu for循环展开的Overhead测试
* src/Makefile 交叉编译文件
* result.pdf 实验结果分析
* result.md 实验结果分析源文件
* pic/ 引用截图或图片
* test_data.xlsx 测试数据统计文件
* README.md 本目录文件
## 源码编译方式
* 1.编译：直接在根目录下运行make指令即可完成编译。
* 2.运行decoding_test与loop_test文件，即可得到运行结果。
  