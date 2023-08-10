import tensorflow as tf
from tensorflow.contrib.layers.python.layers import optimizers
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from utils import *
from model_fn import model_block
from fg_worker import FeatureColumnGenerator
import os
import time
import rtp_fg
import json
import numpy as np


tf.flags.DEFINE_integer('batch_size', 512, 'batch size')
tf.flags.DEFINE_bool('is_training', True, '')
tf.flags.DEFINE_float('keep_prob', 1.0, "")
tf.flags.DEFINE_string('optimizer', 'Adam', "")
tf.flags.DEFINE_integer("task_index", None, "Worker task index")
tf.flags.DEFINE_string("ps_hosts", "", "ps hosts")
tf.flags.DEFINE_string("worker_hosts", "", "worker hosts")
tf.flags.DEFINE_string("job_name", None, "job name: worker or ps")

FLAGS = tf.flags.FLAGS
table_col_num = 6
dnn_parent_scope = "dnn"

def parser(batch, feature_configs):
    columns = batch.get_next()
    room_id, checkin, feature, scene_1_label, scene_2_label, scene_3_label = columns
    feature = tf.reshape(feature, [-1, 1])
    feature = tf.squeeze(feature, axis=1)
    features = rtp_fg.parse_genreated_fg(feature_configs, feature)
    return room_id, checkin, features, scene_1_label, scene_2_label, scene_3_label


def input_fn(files, batch_size, mode, slice_id, slice_count):
    tf.logging.info("slice_count:{}, slice_id:{}".format(slice_count, slice_id))
    if mode == 'train':
        dataset = tf.data.TableRecordDataset(files, [[' ']] * table_col_num, slice_id=slice_id,
                                                 slice_count=slice_count).batch(batch_size)
    return dataset


def model_fn(data_batch, global_step, is_chief):
    # construct the model structure
    tf.logging.info("loading json config...")
    with open('./feature_config.json', 'r') as f:
        feature_configs_java = json.load(f)

    room_id, checkin, features, scene_1_label, scene_2_label, scene_3_label = parser(data_batch, feature_configs_java)
    params = {name: value for name, value in FLAGS.__flags.items()}
    room_id_, checkin_, scene_1_label_, scene_2_label_, scene_3_label_, scene_1_logit, scene_1_logit2, scene_1_logit3, scene_2_logit, scene_2_logit2, scene_2_logit3, scene_3_logit, scene_3_logit2, scene_3_logit3, output_gate = model_block(room_id, checkin, features, scene_1_label, scene_2_label, scene_3_label, fc_generator, FLAGS.is_training, FLAGS.keep_prob, params)
    scene_1_discount = 1.0 + (scene_1_logit2 * tf.pow(scene_1_logit, 1.6) - scene_1_logit3)
    scene_2_discount = 1.0 + (scene_2_logit2 * tf.pow(scene_2_logit, 1.6) - scene_2_logit3)
    scene_3_discount = 1.0 + (scene_3_logit2 * tf.pow(scene_3_logit, 1.6) - scene_3_logit3)


    with tf.name_scope('loss'):
        scene_1_p_loss = tf.losses.log_loss(labels=scene_1_label_, predictions=scene_1_logit)
        scene_1_v_loss = tf.maximum(0., scene_1_label_ + (1 - scene_1_label_) * 0.85 - scene_1_discount) + tf.maximum(0.,
                                                                                                                  scene_1_discount - (
                                                                                                                          1 - scene_1_label_) - 1.15 * scene_1_label_)
        scene_2_p_loss = tf.losses.log_loss(labels=scene_2_label_, predictions=scene_2_logit)
        scene_2_v_loss = tf.maximum(0., scene_2_label_ + (1 - scene_2_label_) * 0.85 - scene_2_discount) + tf.maximum(0.,
                                                                                                              scene_2_discount - (
                                                                                                                      1 - scene_2_label_) - 1.15 * scene_2_label_)
        scene_3_p_loss = tf.losses.log_loss(labels=scene_3_label_, predictions=scene_3_logit)
        scene_3_v_loss = tf.maximum(0., scene_3_label_ + (1 - scene_3_label_) * 0.85 - scene_3_discount) + tf.maximum(0.,
                                                                                                          scene_3_discount - (
                                                                                                                  1 - scene_3_label_) - 1.15 * scene_3_label_)
        loss = 1.0 * (scene_1_p_loss + scene_2_p_loss + scene_3_p_loss) + 0.1 * (scene_1_v_loss + scene_2_v_loss + scene_3_v_loss)
        loss = tf.reduce_sum(loss)
        tf.add_to_collection("losses", loss)
        losses = tf.get_collection('losses')
        tf.logging.info("train losses: {}".format(losses))
        loss_total = tf.add_n(losses)

    train_op = make_training_op(loss_total, global_step)
    return loss_total, train_op, train_metrics


def make_training_op(training_loss, global_step):
    _DNN_LEARNING_RATE = 0.001
    _LINEAR_LEARNING_RATE = 0.005
    _GRADIENT_CLIP_NORM = 100.0
    OPTIMIZER_CLS_NAMES = [
        "Adagrad",
        "Adam",
        "Ftrl",
        "Momentum",
        "RMSProp",
        "SGD"
    ]
    warm_up_learning_rate = 0.0001
    warm_up_step = 50000
    init_learning_rate = 0.001
    decay_steps = 10000
    decay_rate = 0.96
    learning_rate2 = tf.train.exponential_decay(_DNN_LEARNING_RATE,
                                                global_step,
                                                20000,
                                                0.9,
                                                staircase = True
        )
    learning_rate = tf.train.smooth_exponential_decay(warm_up_learning_rate,
                                                      warm_up_step,
                                                      init_learning_rate,
                                                      global_step,
                                                      decay_steps,
                                                      decay_rate)
    # bn
    with ops.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_ops = []
        dnn_optimizer = tf.train.AdamAsyncOptimizer(learning_rate2)
        train_ops.append(
            optimizers.optimize_loss(
                loss=training_loss,
                global_step=global_step,
                learning_rate=_DNN_LEARNING_RATE,
                optimizer=dnn_optimizer,
                variables=ops.get_collection("trainable_variables"),
                name=dnn_parent_scope,
                clip_gradients=None,
                increment_global_step=None))
        train_op = control_flow_ops.group(*train_ops)
        with ops.control_dependencies([train_op]):
            with ops.colocate_with(global_step):
                return state_ops.assign_add(global_step, 1).op


def main(_):
    tf.logging.info("job name = %s" % FLAGS.job_name)
    tf.logging.info("task index = %d" % FLAGS.task_index)
    is_chief = FLAGS.task_index == 0
    ps_spec = FLAGS.ps_hosts.split(",")
    worker_spec = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})
    worker_count = len(worker_spec)
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    # join the ps server
    if FLAGS.job_name == "ps":
        server.join()
    # start the training
    train(worker_count=worker_count, task_index=FLAGS.task_index, cluster=cluster, is_chief=is_chief,
          target=server.target)


if __name__ == '__main__':
    tf.app.run()
