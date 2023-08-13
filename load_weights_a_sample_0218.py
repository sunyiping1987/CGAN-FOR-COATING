import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import matplotlib.pyplot as plt
# %matplotlib inline
import pandas as pd
import numpy as np
import xlrd
import os
import xlwt  # 引用写入的库
import xlsxwriter

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(10000)

tf.__version__


def excel_read_inputs():  # 定义了一个读取excel的函数
    wb1 = xlrd.open_workbook(
        r'F:\现代制造教育部重点实验室课题\博士大论文\大论文图片\第六章\空腔组合三孔腔130\0212吸声系数结果\0212吸声系数结果\训练数据\dataTrain_Sort_Random_X.xlsx')  # 打开Excel文件
    sheet_1 = wb1.sheet_by_name('Sheet1')  # 通过excel表格名称(rank)获取工作表
    names_1 = []  # 创建空list，用来保存人物名称一列
    for a in range(sheet_1.nrows):  # 循环读取表格内容（每次读取一行数据）
        cells_1 = sheet_1.row_values(a)  # 每行数据赋值给cells
        names = cells_1  # 循环读取cell没一列的数据，cell[0]就是包含了每一行的人物名称
        names_1.append(names)  # append这是一个插入函数，功能可以在list之后插入一个name数据
    return names_1


def excel_read_outputs():  # 定义了一个读取excel的函数
    wb2 = xlrd.open_workbook(
        r'F:\现代制造教育部重点实验室课题\博士大论文\大论文图片\第六章\空腔组合三孔腔130\0212吸声系数结果\0212吸声系数结果\训练数据\dataTrain_Sort_Random_Y.xlsx')  # 打开Excel文件
    sheet_2 = wb2.sheet_by_name('Sheet1')  # 通过excel表格名称(rank)获取工作表
    names_2 = []
    for b in range(sheet_2.nrows):  # 循环读取表格内容（每次读取一行数据）
        cells_2 = sheet_2.row_values(b)  # 每行数据赋值给cells
        names = cells_2  # 循环读取cell没一列的数据，cell[0]就是包含了每一行的人物名称
        names_2.append(names)  # append这是一个插入函数，功能可以在list之后插入一个name数据
    return names_2


inputs = excel_read_inputs()  # 这里调用了excel_read的函数，此时a就包含了names、ids、ranks三列数据
inputs = tf.cast(inputs, tf.float32)
inputs_T = tf.transpose(inputs)
print('inputs_T.shape:', inputs_T.shape)
print(tf.reduce_max(inputs_T[0]), tf.reduce_max(inputs_T[1]),
      tf.reduce_max(inputs_T[2]), tf.reduce_max(inputs_T[3]))

outputs = excel_read_outputs()
outputs = tf.cast(outputs, tf.float32)
# outputs = tf.transpose(outputs)
print('outputs.shape:', outputs.shape)

inputs_normal = []
for i in range(6):
    input_i = 2 * (inputs_T[i] - tf.reduce_min(inputs_T[i])) / (
                tf.reduce_max(inputs_T[i]) - tf.reduce_min(inputs_T[i])) - 1
    inputs_normal.append(input_i)
inputs_normal = tf.cast(inputs_normal, dtype=tf.float32)
print('inputs_normal.shape', inputs_normal.shape, tf.reduce_max(inputs_normal[0]), tf.reduce_min(inputs_normal[0]))

inputs_normal = tf.transpose(inputs_normal)
print('inputs_normal_transpose.shape:', inputs_normal.shape)

Epochs = 100 + 1
batchsz = 512
val_step = 10
Lamda = inputs_normal.shape[0]

# db,db_val
train_len = int(0.9 * Lamda)
inputs_normal_train = inputs_normal[:train_len, :]
outputs_train = outputs[:train_len, :]
print(inputs_normal_train.shape)
inputs_normal_val = inputs_normal[train_len:, :]
outputs_val = outputs[train_len:, :]
print(inputs_normal_val.shape)

db = tf.data.Dataset.from_tensor_slices((inputs_normal_train, outputs_train)).shuffle(train_len).batch(batchsz)
db_val = tf.data.Dataset.from_tensor_slices((inputs_normal_val, outputs_val)).batch(batchsz)

db_sample = next(iter(db))
print(db_sample[0].shape, db_sample[1].shape)

db_val_sample = next(iter(db_val))
print(db_val_sample[0].shape, db_val_sample[1].shape)

optimizer = optimizers.Adam(1e-4)

h = 28
w = 28
channel = 1
outputs = 30


def main():
    network = Sequential([

        layers.Dense(1024, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(1024, activation='relu'),

        layers.Dense(outputs, activation='sigmoid')])

    network.build(input_shape=(batchsz, 6))
    network.summary()

    loss_total = []
    loss_val_total = []

    Frequency = range(200, 6200, 200)

    loss_add = 0


    network.load_weights('F:\现代制造教育部重点实验室课题\博士大论文\大论文图片\第六章\空腔组合三孔腔130\程序0218\Paper-main\Paper-main\ckpt\weights.ckpt')
    print('loaded model_weights!')



    def excel_read_inputs_Xtest():  # 定义了一个读取excel的函数
        wb1 = xlrd.open_workbook(
            r'F:\现代制造教育部重点实验室课题\博士大论文\大论文图片\第六章\空腔组合三孔腔130\0212吸声系数结果\0212吸声系数结果\训练数据\XValidation0_normal.xlsx')  # 打开Excel文件
        sheet_1 = wb1.sheet_by_name('Sheet1')  # 通过excel表格名称(rank)获取工作表
        names_1 = []  # 创建空list，用来保存人物名称一列
        for a in range(sheet_1.nrows):  # 循环读取表格内容（每次读取一行数据）
            cells_1 = sheet_1.row_values(a)  # 每行数据赋值给cells
            names = cells_1  # 循环读取cell没一列的数据，cell[0]就是包含了每一行的人物名称
            names_1.append(names)  # append这是一个插入函数，功能可以在list之后插入一个name数据
        return names_1

    x_test = excel_read_inputs_Xtest()  # 这里调用了excel_read的函数，此时a就包含了names、ids、ranks三列数据
    x_test = tf.cast(x_test, tf.float32)
    # x_test = (x_test + 1) / 2
    print('x_test:', x_test.shape)

    def excel_read_inputs_Ytest():  # 定义了一个读取excel的函数
        wb1 = xlrd.open_workbook(
            r'F:\现代制造教育部重点实验室课题\博士大论文\大论文图片\第六章\空腔组合三孔腔130\0212吸声系数结果\0212吸声系数结果\训练数据\YValidation0.xlsx')  # 打开Excel文件
        sheet_1 = wb1.sheet_by_name('Sheet1')  # 通过excel表格名称(rank)获取工作表
        names_1 = []  # 创建空list，用来保存人物名称一列
        for a in range(sheet_1.nrows):  # 循环读取表格内容（每次读取一行数据）
            cells_1 = sheet_1.row_values(a)  # 每行数据赋值给cells
            names = cells_1  # 循环读取cell没一列的数据，cell[0]就是包含了每一行的人物名称
            names_1.append(names)  # append这是一个插入函数，功能可以在list之后插入一个name数据
        return names_1

    y_test = excel_read_inputs_Ytest()  # 这里调用了excel_read的函数，此时a就包含了names、ids、ranks三列数据
    y_test = tf.cast(y_test, tf.float32)
    print('y_test:', y_test.shape)

    # print(x_train.shape, y_train.shape)
    print('x_test.shape, y_test.shape:', x_test.shape, y_test.shape)



    out_last = network(x_test)
    print('predicted', out_last.shape)


    # out_last 保存
    workbook = xlwt.Workbook()
    worksheet = workbook.add_sheet("out_last_saved")  # 新建sheet
    row, col = 0, 0

    for item, cost in enumerate(out_last):
        col = 0
        for i in range(30):
            # print('item:', item, 'cost', cost)
            worksheet.write(item, col, np.float(cost[i]))
            col += 1
        row += 1
    workbook.save(
        r'F:\现代制造教育部重点实验室课题\博士大论文\大论文图片\第六章\空腔组合三孔腔130\0212吸声系数结果\0212吸声系数结果\训练数据\predicted_inverse_tensorflow_news_0218.xls')  # 保存

    print('out_last_saved_tensorflow:', out_last.shape)




if __name__ == '__main__':
    main()
