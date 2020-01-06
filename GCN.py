# encoding:utf-8
import hdfs
import tensorflow as tf
import numpy as np
import time
from module import node_classification
from module import traning_data, testing_data

flags = tf.app.flags
# 定义数据
flags.DEFINE_string('data_dir', '/graph/', 'Directory for storing graph data')
flags.DEFINE_string('method', 'GCN', 'Method')
flags.DEFINE_integer('node_num', None, 'The number of nodes')
flags.DEFINE_integer('labels', 42, 'The number of node types')

# 定义默认训练参数
flags.DEFINE_integer('dim', 128, 'Dimension of each embedding')
flags.DEFINE_integer('train_steps', 1000, 'Number of training steps to perform')
# flags.DEFINE_integer('batch_size', 64, 'Training batch size ')
flags.DEFINE_integer('thread', 2, 'Depend on  memory size, the number of parallel walks')
flags.DEFINE_float('semi_size', 0.8, 'Testing batch size ')
flags.DEFINE_float('lr', 0.001, 'Learning rate')

# 定义aggregate参数
flags.DEFINE_integer('k', 5, 'The aggregated neighbors for each node')

# 定义分布式参数
flags.DEFINE_integer('worker', None, 'The number of worker')

# 参数服务器parameter server节点和计算服务器worker节点信息
cluster_dic = {
    "worker": [
        "10.76.3.92:2223", # worker节点地址：端口
        "10.76.3.110:2224",
        "10.76.3.89:2225",
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
    # FLAGS.train_steps = 2
    is_chief = (FLAGS.task == 0)

    if FLAGS.job == 'ps':
        server.join()

    with tf.device(tf.train.replica_device_setter(cluster=cluster)):
        global_step = tf.Variable(0, name='global_step', trainable=False)  # 创建纪录全局训练步数变量

        emb_init = (np.random.randn(FLAGS.node_num, FLAGS.dim) / np.sqrt(FLAGS.node_num / 2)).astype('float32')
        W_init = (np.random.randn(FLAGS.dim, FLAGS.labels) / np.sqrt(FLAGS.dim / 2)).astype('float32')
        emb = tf.Variable(emb_init, name='emb', trainable=True) # 创建embedding向量并且初始化

        split = FLAGS.node_num // FLAGS.worker
        span = int(FLAGS.semi_size*split)
        # 训练参数占位符
        xa = tf.placeholder(dtype=tf.int32, shape=(split, FLAGS.k*FLAGS.k))# aggregation的集合
        xs = tf.placeholder(dtype=tf.int32, shape=(int(split*FLAGS.semi_size))) # 有label的点计算loss，半监督
        ys = tf.placeholder(dtype=tf.int32, shape=(int(split*FLAGS.semi_size), FLAGS.labels))# label
        W = tf.Variable(W_init, name='weight', trainable=True)# 权重矩阵

        # 测试参数占位符
        x = tf.placeholder(dtype=tf.int32, shape=(FLAGS.worker*span))
        x_label = tf.placeholder(dtype=tf.int32, shape=(FLAGS.worker*span))
        y = tf.placeholder(dtype=tf.int32, shape=(FLAGS.worker*(split - span)))
        y_label = tf.placeholder(dtype=tf.int32, shape=(FLAGS.worker*(split - span)))

        x_e = tf.reduce_sum(tf.squeeze(tf.nn.embedding_lookup(emb, xa)), axis=1) # (batch_size, dim)

        x_l = tf.squeeze(tf.nn.embedding_lookup(x_e, xs)) # (split*semi_size, dim)

        logits = tf.nn.relu(tf.matmul(x_l, W))

        # loss用softmax作多分类
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ys))

        optimizer = tf.train.AdamOptimizer(FLAGS.lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        f1_score = tf.py_func(node_classification, [x, x_label, y, y_label, emb], tf.double, stateful=True)

        init_op = tf.global_variables_initializer()# 参数初始化
        sv = tf.train.Supervisor(is_chief=is_chief, init_op=init_op, recovery_wait_secs=1,
                                 global_step=global_step)

        if is_chief:
            print('Worker %d: Initailizing session...' % FLAGS.task)
        else:
            print('Worker %d: Waiting for session to be initaialized...' % FLAGS.task)

        # 同步开始
        sess = sv.prepare_or_wait_for_session(server.target)
        print('Worker %d: Session initialization  complete.' % FLAGS.task)

        time_begin = time.time()
        print('Traing begins @ %f' % time_begin)

        local_step = 0
        step = 0
        dx, dx_label, dy, dy_label = testing_data(FLAGS, G)
        while not sv.should_stop() and step <= FLAGS.train_steps:
            dxa, dxs, dys = traning_data(FLAGS, G, local_step)
            train_feed = {
                xa:dxa,
                xs:dxs,
                ys:dys
            }
            _, step, _loss = sess.run([train_op, global_step, loss], feed_dict=train_feed)
            local_step += 1

            now = time.time()
            print('%f: Worker %d: traing step %d dome (global step:%d/%d), and loss : %f' % (now, FLAGS.task, local_step-1, step, FLAGS.train_steps, _loss))

            if local_step%10==0 and local_step!=0:
                f1_ = sess.run([f1_score], feed_dict={
                    x: dx,
                    x_label: dx_label,
                    y: dy,
                    y_label: dy_label
                })[0]
                print("Node classification macro-f1 is %.2f, micro-f1 is %.2lf" % (f1_[0], f1_[1]))

            if step >= FLAGS.train_steps:
                break
        f1_ = sess.run([f1_score], feed_dict={
            x:dx,
            x_label:dx_label,
            y:dy,
            y_label:dy_label
        })[0]
        print("Node classification macro-f1 is %.2f, micro-f1 is %.2lf" % (f1_[0], f1_[1]))
        time_end = time.time()
        print('Training ends @ %f' % time_end)
        train_time = time_end - time_begin
        print('Training elapsed time:%f s' % train_time)

        sleep_time = 0
        while sleep_time < 5:
            time.sleep(2)
            sleep_time += 1
            print("Waiting other machines...")
    sess.close()

if __name__ == '__main__':
    tf.app.run()