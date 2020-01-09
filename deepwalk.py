# encoding:utf-8
import hdfs
import tensorflow as tf
import numpy as np
import time
from module import link_prediction
from module import traning_data, testing_data


flags = tf.app.flags
# 定义数据
flags.DEFINE_string('data_dir', '/graph/', 'Directory for storing graph data')
flags.DEFINE_string('method', 'deepwalk', 'Method')
flags.DEFINE_integer('node_num', None, 'The number of nodes')

# 定义默认训练参数
flags.DEFINE_integer('dim', 128, 'Dimension of each embedding')
flags.DEFINE_integer('train_steps', 10000, 'Number of training steps to perform')
flags.DEFINE_integer('batch_size', 64, 'Training batch size ')
flags.DEFINE_integer('test_size', 10000, 'Testing batch size ')
flags.DEFINE_integer('thread', 2, 'Depend on  memory size, the number of parallel walks')
flags.DEFINE_float('lr', 0.01, 'Learning rate')

# 定义随机游走参数
flags.DEFINE_integer('w', 40, 'The max length of random walk at each node')
flags.DEFINE_integer('ns', 6, 'Negative samplers')
flags.DEFINE_integer('cs', 2, 'Context size')

# 定义分布式参数
flags.DEFINE_integer('worker', None, 'The number of worker')

# 参数服务器parameter server节点和计算服务器worker节点信息
cluster_dic = {
    "worker": [
        "10.76.3.92:2223", # worker节点地址：端口
        "10.76.3.92:2224",
        # "10.76.3.89:2225",
    ],
    "ps": [
        "10.76.3.92:2222"  # ps节点地址：端口
    ]}

cluster = tf.train.ClusterSpec(cluster_dic)
# 设置job name参数
flags.DEFINE_string('job', None, 'job name: worker or ps')
# 设置任务的索引
flags.DEFINE_integer('task', None, 'Index of task within the job')

FLAGS = flags.FLAGS

def main(args):
    server = tf.train.Server(cluster, job_name=FLAGS.job, task_index=FLAGS.task)

    client = hdfs.Client("http://10.76.3.92:50070", root='/', timeout=100)
    with client.read('/graph.txt') as reader:
        G = eval(reader.read())
    FLAGS.worker = len(cluster_dic['worker'])
    FLAGS.node_num = len(G.keys()) - 1

    # FLAGS.train_steps = FLAGS.node_num//FLAGS.batch_size*20
    FLAGS.train_steps = 4
    is_chief = (FLAGS.task == 0)

    if FLAGS.job == 'ps':
        server.join()

    with tf.device(tf.train.replica_device_setter(cluster=cluster)):
        global_step = tf.Variable(0, name='global_step', trainable=False)  # 创建纪录全局训练步数变量
        emb_init = (np.random.randn(FLAGS.node_num, FLAGS.dim) / np.sqrt(FLAGS.node_num / 2)).astype('float32')
        emb = tf.Variable(emb_init, name='emb', trainable=True) # 创建embedding向量并且初始化

        L_con = 0
        L_ucon = 0

        # 训练参数占位符
        pos = (FLAGS.w-2*FLAGS.cs)*2*FLAGS.cs

        xc_0 = tf.placeholder(dtype=tf.int32, shape=(pos * FLAGS.batch_size))# 正边 source
        xc_1 = tf.placeholder(dtype=tf.int32, shape=(pos * FLAGS.batch_size))# 正边 target
        xuc_0 = tf.placeholder(dtype=tf.int32, shape=(pos * FLAGS.ns * FLAGS.batch_size))# 无边 source
        xuc_1 = tf.placeholder(dtype=tf.int32, shape=(pos * FLAGS.ns * FLAGS.batch_size))# 无边 target

        # 测试参数占位符
        val = tf.placeholder(dtype=tf.int32, shape=(2, FLAGS.test_size, 2))# 随机抽取正边 = 负边

        # 将边序列映射到embedding上
        con_0_emb = tf.squeeze(tf.nn.embedding_lookup(emb, xc_0))  # (batch,  dim)
        con_1_emb = tf.squeeze(tf.nn.embedding_lookup(emb, xc_1))  # (batch,  dim)
        ucon_0_emb = tf.squeeze(tf.nn.embedding_lookup(emb, xuc_0))  # (batch,  dim)
        ucon_1_emb = tf.squeeze(tf.nn.embedding_lookup(emb, xuc_1))  # (batch,  dim)

        # 计算边相似度是，包括positive samples 和 negative samples
        con_v = tf.sigmoid(tf.squeeze(tf.reduce_sum(tf.einsum('ni,ni->ni', con_0_emb, con_1_emb), -1)))
        ucon_v = tf.sigmoid(tf.squeeze(tf.reduce_sum(tf.einsum('ni,ni->ni', ucon_0_emb, ucon_1_emb), -1)))

        # 计算skip-gram的loss
        L_con -= tf.reduce_sum(tf.log(con_v + 1e-15))  # connection
        L_ucon -= tf.reduce_sum(tf.log(1 - ucon_v + 1e-15))  # unconnection
        loss = (L_con + L_ucon)

        optimizer = tf.train.AdamOptimizer(FLAGS.lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        AUC = tf.py_func(link_prediction, [val, emb], tf.double, stateful=True)

        # init_op = tf.global_variables_initializer()# 参数初始化
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=is_chief,
                                              # hooks=[tf.train.StopAtStepHook(last_step=FLAGS.train_steps),
                                              #                 tf.train.NanTensorHook(loss)],
                                              #  checkpoint_dir="./checkpoint_dir",
                                               save_checkpoint_steps=100) as sess:
            time_begin = time.time()
            print('Traing begins @ %f' % time_begin)

            local_step = 0
            step = 0

            dval = testing_data(FLAGS, G)
            val_feed = {
                val: dval
            }
            while not sess.should_stop() and step <= FLAGS.train_steps:
                dxc_0, dxc_1, dxuc_0, dxuc_1 = traning_data(FLAGS, G, local_step)
                train_feed = {
                    xc_0:dxc_0,
                    xc_1:dxc_1,
                    xuc_0:dxuc_0,
                    xuc_1:dxuc_1
                }

                _, step, _loss = sess.run([train_op, global_step, loss], feed_dict=train_feed)
                local_step += 1

                now = time.time()
                print('%f: Worker %d: traing step %d dome (global step:%d/%d), and loss : %f' % (now, FLAGS.task, local_step-1, step, FLAGS.train_steps, _loss))

                if local_step%10==0 and local_step!=0:
                    auc = sess.run([AUC], feed_dict=val_feed)
                    print("Link prediction AUC is %.2f" % auc[0])

                if step >= FLAGS.train_steps:
                    break

            auc = sess.run([AUC], feed_dict = val_feed)
            print("Link prediction AUC is %.2f" % auc[0])
            time_end = time.time()
            print('Training ends @ %f' % time_end)
            train_time = time_end - time_begin
            print('Training elapsed time:%f s' % train_time)

            sleep_time = 0
            while sleep_time < 5:
                time.sleep(2)
                sleep_time += 1
                print("Waiting other machines...")

if __name__ == '__main__':
    tf.app.run()