# CNN_power_load_prediction_model
I try to make a hotspot for the electricity load dataset to gain the unknown features of the time series data, and use CNN to fit the predictive model. 
# 基于CNN和网格搜索交叉验证的电力负荷预测

本项目旨在使用卷积神经网络（CNN）模型根据CSV文件中提供的特征预测电力负荷。CSV文件包含各种气象和电力相关数据。生成热点图以可视化不同特征之间的相关性。然后使用CNN模型根据提供的特征预测电力负荷。实现网格搜索和交叉验证以找到CNN模型的最佳超参数。

## 依赖项

- Python 3.7 或更高版本
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- TensorFlow 2.x

您可以使用以下命令安装所需的库：


## 使用方法

1. 在代码中将 '电力.csv' 占位符替换为实际的CSV文件路径。
2. 运行提供的Python脚本以预处理数据、生成热点图、创建CNN模型、执行带交叉验证的网格搜索、训练优化模型以及进行预测：


3. 生成的热点图将保存为 'hotspot_graph.png'，位于工作目录中。

4. 可以使用训练好的CNN模型根据输入特征预测电力负荷。

## 定制化

- 您可以在`create_cnn_with_params`函数中修改CNN架构或超参数。
- 调整网格搜索交叉验证中的参数网格，即`param_grid`变量。
- 若要提高模型性能，可以考虑使用其他深度学习模型，如LSTM或GRU（如果涉及时间序列数据）。
- 请随意尝试不同的预处理技巧、特征工程或数据增强以提高模型性能。
