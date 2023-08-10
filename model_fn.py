
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.layers.python.layers import optimizers
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops
from utils import *
from fg_worker import FeatureColumnGenerator

import math
import os
import time
import json
import copy


def model_block(room_id, checkin, features, scene_1_label, scene_2_label, scene_3_label, fc_generator, is_training, keep_prob, params):

    outputs_dict = fc_generator.get_output_dict(features)
    for key in outputs_dict:
        tf.logging.info(key)
        tf.logging.info(outputs_dict[key])

    scene_1_feats = []
    scene_2_feats = []
    scene_3_feats = []
    comm_feats = []
    simi_query = []
    day_query = []
    for key in outputs_dict:
        if key in ["scene_1_1", "scene_1_2", "scene_1_3", "scene_1_4"]:
            scene_1_feats.append((key, outputs_dict[key]))
        if key in ["scene_2_1","scene_2_2","scene_2_3","scene_2_4"]:
            scene_2_feats.append((key, outputs_dict[key]))
        if key in ["scene_3_1","scene_3_2","scene_3_3","scene_3_4"]:
            scene_3_feats.append((key, outputs_dict[key]))
        if key in ["comm_1","comm_2","comm_3","comm_4","comm_5","comm_6","comm_7","comm_8","comm_9","comm_10","comm_11","comm_12","comm_13","comm_14","comm_15","comm_16","comm_17"]:
            comm_feats.append((key, outputs_dict[key]))
        if key in ["comm_1","comm_3","comm_4","comm_6","comm_9","comm_10","comm_11","comm_12","comm_13","comm_14"]:
            simi_query.append((key, outputs_dict[key]))
        if key in ["comm_15","comm_16"]:
            day_query.append((key, outputs_dict[key]))
    scene_2_feats = [feat for _, feat in sorted(scene_2_feats, key=lambda x: x[0])]
    scene_3_feats = [feat for _, feat in sorted(scene_3_feats, key=lambda x: x[0])]
    scene_1_feats = [feat for _, feat in sorted(scene_1_feats, key=lambda x: x[0])]
    comm_feats = [feat for _, feat in sorted(comm_feats, key=lambda x: x[0])]
    simi_query = [feat for _, feat in sorted(simi_query, key=lambda x: x[0])]
    day_query = [feat for _, feat in sorted(day_query, key=lambda x: x[0])]

    scene_2_feats = tf.concat(scene_2_feats, axis=1)
    scene_3_feats = tf.concat(scene_3_feats, axis=1)
    scene_1_feats = tf.concat(scene_1_feats, axis=1)
    comm_feats = tf.concat(comm_feats, axis=1)
    simi_query = tf.concat(simi_query, axis=1)
    day_query = tf.concat(day_query, axis=1)

    # source sequence of demand
    scene_1_rid_num_list = outputs_dict["scene_1_rid_num_list"]
    scene_1_ord_uv_list = outputs_dict["scene_1_ord_uv_list"]
    scene_1_jy_list = outputs_dict["scene_1_jy_list"]
    room_scene_1_ord_uv_list = outputs_dict["room_scene_1_ord_uv_list"]
    room_scene_1_jy_list = outputs_dict["room_scene_1_jy_list"]
    scene_2_rid_num_list = outputs_dict["scene_2_rid_num_list"]
    scene_2_ord_uv_list = outputs_dict["scene_2_ord_uv_list"]
    scene_2_jy_list = outputs_dict["scene_2_jy_list"]
    room_scene_2_ord_uv_list = outputs_dict["room_scene_2_ord_uv_list"]
    room_scene_2_jy_list = outputs_dict["room_scene_2_jy_list"]
    scene_3_rid_num_list = outputs_dict["scene_3_rid_num_list"]
    scene_3_ord_uv_list = outputs_dict["scene_3_ord_uv_list"]
    scene_3_jy_list = outputs_dict["scene_3_jy_list"]
    room_scene_3_ord_uv_list = outputs_dict["room_scene_3_ord_uv_list"]
    room_scene_3_jy_list = outputs_dict["room_scene_3_jy_list"]

    ipv_uv_list = outputs_dict["ipv_uv_list"]
    search_uv_list = outputs_dict["search_uv_list"]
    sale_price_list = outputs_dict["sale_price_list"]
    room_ipv_uv_list = outputs_dict["room_ipv_uv_list"]
    room_search_uv_list = outputs_dict["room_search_uv_list"]

    compete_mask = outputs_dict["compete_mask"]
    scene_2_compete_mask = outputs_dict["scene_2_compete_mask"]
    scene_1_price_mask = outputs_dict["scene_1_price_mask"]
    scene_3_price_mask = outputs_dict["scene_3_price_mask"]

    scene_1_rid_num_list = tf.expand_dims(scene_1_rid_num_list, axis=2)  # None*30*1
    scene_1_ord_uv_list = tf.expand_dims(scene_1_ord_uv_list, axis=2)
    scene_1_jy_list = tf.expand_dims(scene_1_jy_list, axis=2)
    room_scene_1_ord_uv_list = tf.expand_dims(room_scene_1_ord_uv_list, axis=2)
    room_scene_1_jy_list = tf.expand_dims(room_scene_1_jy_list, axis=2)
    scene_2_rid_num_list = tf.expand_dims(scene_2_rid_num_list, axis=2)
    scene_2_ord_uv_list = tf.expand_dims(scene_2_ord_uv_list, axis=2)
    scene_2_jy_list = tf.expand_dims(scene_2_jy_list, axis=2)
    room_scene_2_ord_uv_list = tf.expand_dims(room_scene_2_ord_uv_list, axis=2)
    room_scene_2_jy_list = tf.expand_dims(room_scene_2_jy_list, axis=2)
    scene_3_rid_num_list = tf.expand_dims(scene_3_rid_num_list, axis=2)
    scene_3_ord_uv_list = tf.expand_dims(scene_3_ord_uv_list, axis=2)
    scene_3_jy_list = tf.expand_dims(scene_3_jy_list, axis=2)
    room_scene_3_ord_uv_list = tf.expand_dims(room_scene_3_ord_uv_list, axis=2)
    room_scene_3_jy_list = tf.expand_dims(room_scene_3_jy_list, axis=2)

    ipv_uv_list = tf.expand_dims(ipv_uv_list, axis=2)
    search_uv_list = tf.expand_dims(search_uv_list, axis=2)
    sale_price_list = tf.expand_dims(sale_price_list, axis=2)
    room_ipv_uv_list = tf.expand_dims(room_ipv_uv_list, axis=2)
    room_search_uv_list = tf.expand_dims(room_search_uv_list, axis=2)

    compete_mask = tf.expand_dims(compete_mask, axis=2)   #None*30*1
    scene_2_compete_mask = tf.expand_dims(scene_2_compete_mask, axis=2)
    scene_1_price_mask = tf.expand_dims(scene_1_price_mask, axis=2)
    scene_3_price_mask = tf.expand_dims(scene_3_price_mask, axis=2)

    # sequence embedding
    scene_1_ord_price_rate_seq = outputs_dict["shared_scene_1_ord_price_rate_seq"]
    scene_1_ord_price_rate_seq = tf.concat([tf.expand_dims(id, axis=1) for id in scene_1_ord_price_rate_seq], axis=1)   #None*30*8

    scene_1_room_ord_price_rate_seq = outputs_dict["shared_scene_1_room_ord_price_rate_seq"]
    scene_1_room_ord_price_rate_seq = tf.concat([tf.expand_dims(id, axis=1) for id in scene_1_room_ord_price_rate_seq], axis=1)

    scene_2_ord_price_rate_seq = outputs_dict["shared_scene_2_ord_price_rate_seq"]
    scene_2_ord_price_rate_seq = tf.concat([tf.expand_dims(id, axis=1) for id in scene_2_ord_price_rate_seq], axis=1)

    scene_2_room_ord_price_rate_seq = outputs_dict["shared_scene_2_room_ord_price_rate_seq"]
    scene_2_room_ord_price_rate_seq = tf.concat([tf.expand_dims(id, axis=1) for id in scene_2_room_ord_price_rate_seq], axis=1)

    scene_3_ord_price_rate_seq = outputs_dict["shared_scene_3_ord_price_rate_seq"]
    scene_3_ord_price_rate_seq = tf.concat([tf.expand_dims(id, axis=1) for id in scene_3_ord_price_rate_seq], axis=1)

    scene_3_room_ord_price_rate_seq = outputs_dict["shared_scene_3_room_ord_price_rate_seq"]
    scene_3_room_ord_price_rate_seq = tf.concat([tf.expand_dims(id, axis=1) for id in scene_3_room_ord_price_rate_seq], axis=1)

    scene_1_med_compt_price_seq = outputs_dict["shared_scene_1_med_compt_price_seq"]
    scene_1_med_compt_price_seq = tf.concat([tf.expand_dims(id, axis=1) for id in scene_1_med_compt_price_seq], axis=1)
    scene_1_med_compt_price_seq = tf.multiply(scene_1_med_compt_price_seq, compete_mask)    

    scene_1_min_compt_price_seq = outputs_dict["shared_scene_1_min_compt_price_seq"]
    scene_1_min_compt_price_seq = tf.concat([tf.expand_dims(id, axis=1) for id in scene_1_min_compt_price_seq], axis=1)
    scene_1_min_compt_price_seq = tf.multiply(scene_1_min_compt_price_seq, compete_mask)

    scene_2_med_compt_price_seq = outputs_dict["shared_scene_2_med_compt_price_seq"]
    scene_2_med_compt_price_seq = tf.concat([tf.expand_dims(id, axis=1) for id in scene_2_med_compt_price_seq], axis=1)
    scene_2_med_compt_price_seq = tf.multiply(scene_2_med_compt_price_seq, compete_mask)

    scene_2_min_compt_price_seq = outputs_dict["shared_scene_2_min_compt_price_seq"]
    scene_2_min_compt_price_seq = tf.concat([tf.expand_dims(id, axis=1) for id in scene_2_min_compt_price_seq], axis=1)
    scene_2_min_compt_price_seq = tf.multiply(scene_2_min_compt_price_seq, compete_mask)

    scene_3_med_compt_price_seq = outputs_dict["shared_scene_3_med_compt_price_seq"]
    scene_3_med_compt_price_seq = tf.concat([tf.expand_dims(id, axis=1) for id in scene_3_med_compt_price_seq], axis=1)
    scene_3_med_compt_price_seq = tf.multiply(scene_3_med_compt_price_seq, compete_mask)

    scene_3_min_compt_price_seq = outputs_dict["shared_scene_3_min_compt_price_seq"]
    scene_3_min_compt_price_seq = tf.concat([tf.expand_dims(id, axis=1) for id in scene_3_min_compt_price_seq], axis=1)
    scene_3_min_compt_price_seq = tf.multiply(scene_3_min_compt_price_seq, compete_mask)


    week_id_seq = outputs_dict["shared_comm_15"]
    week_id_seq = tf.concat([tf.expand_dims(id, axis=1) for id in week_id_seq], axis=1)

    holid_days_seq = outputs_dict["shared_comm_16"]
    holid_days_seq = tf.concat([tf.expand_dims(id, axis=1) for id in holid_days_seq], axis=1)

    scene_1_med_compt_price_scene_2_seq = outputs_dict["shared_scene_1_med_compt_price_scene_2_seq"]
    scene_1_med_compt_price_scene_2_seq = tf.concat([tf.expand_dims(id, axis=1) for id in scene_1_med_compt_price_scene_2_seq], axis=1)
    scene_1_med_compt_price_scene_2_seq = tf.multiply(scene_1_med_compt_price_scene_2_seq, scene_2_compete_mask)

    scene_1_min_compt_price_scene_2_seq = outputs_dict["shared_scene_1_min_compt_price_scene_2_seq"]
    scene_1_min_compt_price_scene_2_seq = tf.concat([tf.expand_dims(id, axis=1) for id in scene_1_min_compt_price_scene_2_seq], axis=1)
    scene_1_min_compt_price_scene_2_seq = tf.multiply(scene_1_min_compt_price_scene_2_seq, scene_2_compete_mask)

    scene_2_med_compt_price_scene_2_seq = outputs_dict["shared_scene_2_med_compt_price_scene_2_seq"]
    scene_2_med_compt_price_scene_2_seq = tf.concat([tf.expand_dims(id, axis=1) for id in scene_2_med_compt_price_scene_2_seq], axis=1)
    scene_2_med_compt_price_scene_2_seq = tf.multiply(scene_2_med_compt_price_scene_2_seq, scene_2_compete_mask)

    scene_2_min_compt_price_scene_2_seq = outputs_dict["shared_scene_2_min_compt_price_scene_2_seq"]
    scene_2_min_compt_price_scene_2_seq = tf.concat([tf.expand_dims(id, axis=1) for id in scene_2_min_compt_price_scene_2_seq], axis=1)
    scene_2_min_compt_price_scene_2_seq = tf.multiply(scene_2_min_compt_price_scene_2_seq, scene_2_compete_mask)

    scene_3_med_compt_price_scene_2_seq = outputs_dict["shared_scene_3_med_compt_price_scene_2_seq"]
    scene_3_med_compt_price_scene_2_seq = tf.concat([tf.expand_dims(id, axis=1) for id in scene_3_med_compt_price_scene_2_seq], axis=1)
    scene_3_med_compt_price_scene_2_seq = tf.multiply(scene_3_med_compt_price_scene_2_seq, scene_2_compete_mask)

    scene_3_min_compt_price_scene_2_seq = outputs_dict["shared_scene_3_min_compt_price_scene_2_seq"]
    scene_3_min_compt_price_scene_2_seq = tf.concat([tf.expand_dims(id, axis=1) for id in scene_3_min_compt_price_scene_2_seq], axis=1)
    scene_3_min_compt_price_scene_2_seq = tf.multiply(scene_3_min_compt_price_scene_2_seq, scene_2_compete_mask)

    scene_1_simi_price_rate_seq = outputs_dict["shared_scene_1_simi_price_rate_seq"]
    scene_1_simi_price_rate_seq = tf.concat([tf.expand_dims(id, axis=1) for id in scene_1_simi_price_rate_seq], axis=1)
    scene_1_simi_price_rate_seq = tf.multiply(scene_1_simi_price_rate_seq, scene_1_price_mask)

    scene_2_simi_price_rate_seq = outputs_dict["shared_scene_2_simi_price_rate_seq"]
    scene_2_simi_price_rate_seq = tf.concat([tf.expand_dims(id, axis=1) for id in scene_2_simi_price_rate_seq], axis=1)
    scene_2_simi_price_rate_seq = tf.multiply(scene_2_simi_price_rate_seq, scene_1_price_mask)

    scene_3_simi_price_rate_seq = outputs_dict["shared_scene_3_simi_price_rate_seq"]
    scene_3_simi_price_rate_seq = tf.concat([tf.expand_dims(id, axis=1) for id in scene_3_simi_price_rate_seq], axis=1)
    scene_3_simi_price_rate_seq = tf.multiply(scene_3_simi_price_rate_seq, scene_3_price_mask)

    simi_bed_size_seq = outputs_dict["shared_comm_11"]
    simi_bed_size_seq = tf.concat([tf.expand_dims(id, axis=1) for id in simi_bed_size_seq], axis=1)

    simi_max_occupancy_seq = outputs_dict["shared_comm_12"]
    simi_max_occupancy_seq = tf.concat([tf.expand_dims(id, axis=1) for id in simi_max_occupancy_seq], axis=1)

    simi_room_area_seq = outputs_dict["shared_comm_13"]
    simi_room_area_seq = tf.concat([tf.expand_dims(id, axis=1) for id in simi_room_area_seq], axis=1)

    simi_window_type_seq = outputs_dict["shared_comm_14"]
    simi_window_type_seq = tf.concat([tf.expand_dims(id, axis=1) for id in simi_window_type_seq], axis=1)

    simi_business_id_seq = outputs_dict["shared_simi_business_id_seq"]
    simi_business_id_seq = tf.concat([tf.expand_dims(id, axis=1) for id in simi_business_id_seq], axis=1)

    simi_brand_type_seq = outputs_dict["shared_simi_brand_type_seq"]
    simi_brand_type_seq = tf.concat([tf.expand_dims(id, axis=1) for id in simi_brand_type_seq], axis=1)

    simi_star_seq = outputs_dict["shared_comm_4"]
    simi_star_seq = tf.concat([tf.expand_dims(id, axis=1) for id in simi_star_seq], axis=1)

    simi_city_seq = outputs_dict["shared_simi_city_seq"]
    simi_city_seq = tf.concat([tf.expand_dims(id, axis=1) for id in simi_city_seq], axis=1)

    simi_has_park_seq = outputs_dict["shared_comm_9"]
    simi_has_park_seq = tf.concat([tf.expand_dims(id, axis=1) for id in simi_has_park_seq], axis=1)

    simi_is_credit_htl_seq = outputs_dict["shared_comm_10"]
    simi_is_credit_htl_seq = tf.concat([tf.expand_dims(id, axis=1) for id in simi_is_credit_htl_seq], axis=1)


    ## part of demand
    scene_1_demand_series = tf.concat(
        [scene_1_rid_num_list, scene_1_ord_uv_list, scene_1_jy_list, room_scene_1_ord_uv_list, room_scene_1_jy_list, ipv_uv_list, search_uv_list,
         sale_price_list, room_ipv_uv_list, room_search_uv_list], axis=2)
    scene_2_demand_series = tf.concat(
        [scene_2_rid_num_list, scene_2_ord_uv_list, scene_2_jy_list, room_scene_2_ord_uv_list, room_scene_2_jy_list, ipv_uv_list,
         search_uv_list, sale_price_list, room_ipv_uv_list, room_search_uv_list], axis=2)
    scene_3_demand_series = tf.concat(
        [scene_3_rid_num_list, scene_3_ord_uv_list, scene_3_jy_list, room_scene_3_ord_uv_list, room_scene_3_jy_list, ipv_uv_list,
         search_uv_list, sale_price_list, room_ipv_uv_list, room_search_uv_list], axis=2)
    ### RNN
    scene_1_demand_output = tf.keras.layers.GRU(128)(scene_1_demand_series) 
    scene_2_demand_output = tf.keras.layers.GRU(128)(scene_2_demand_series)
    scene_3_demand_output = tf.keras.layers.GRU(128)(scene_3_demand_series)

    demand_input = [scene_1_demand_output, scene_2_demand_output, scene_3_demand_output, tf.concat([scene_1_demand_output, scene_2_demand_output, scene_3_demand_output], axis=1)]
    PLE_output, output_gate = PLE_emb(demand_input, 3, 3, 3, 128, 2)
    scene_1_demand_output = PLE_output[0]
    scene_2_demand_output = PLE_output[1]
    scene_3_demand_output = PLE_output[2]


    ### part of price competitiveness
    day_key = tf.concat([week_id_seq, holid_days_seq], axis=2)  ### None*30*16
    day_query = tf.expand_dims(day_query, axis=2)  ### None*16*1
    day_att_weight = tf.nn.softmax(tf.matmul(day_key, day_query))  # None*30*1

    simi_room_key = tf.concat(
        [simi_business_id_seq, simi_brand_type_seq, simi_star_seq, simi_city_seq, simi_has_park_seq,
         simi_is_credit_htl_seq, simi_bed_size_seq, simi_max_occupancy_seq, simi_room_area_seq, simi_window_type_seq],
        axis=2)
    simi_query = tf.expand_dims(simi_query, axis=2)  # None*80*1
    simi_att_weight = tf.nn.softmax(tf.matmul(simi_room_key, simi_query))  # None*5*1

    scene_1_ord_price_rate_seq = tf.reduce_mean(scene_1_ord_price_rate_seq, axis=1)  # None*8
    scene_1_room_ord_price_rate_seq = tf.reduce_mean(scene_1_room_ord_price_rate_seq, axis=1)
    scene_2_ord_price_rate_seq = tf.reduce_mean(scene_2_ord_price_rate_seq, axis=1)
    scene_2_room_ord_price_rate_seq = tf.reduce_mean(scene_2_room_ord_price_rate_seq, axis=1)
    scene_3_ord_price_rate_seq = tf.reduce_mean(scene_3_ord_price_rate_seq, axis=1)
    scene_3_room_ord_price_rate_seq = tf.reduce_mean(scene_3_room_ord_price_rate_seq, axis=1)
    scene_1_med_compt_price_scene_2_seq = tf.reduce_mean(scene_1_med_compt_price_scene_2_seq, axis=1)
    scene_1_min_compt_price_scene_2_seq = tf.reduce_mean(scene_1_min_compt_price_scene_2_seq, axis=1)
    scene_2_med_compt_price_scene_2_seq = tf.reduce_mean(scene_2_med_compt_price_scene_2_seq, axis=1)
    scene_2_min_compt_price_scene_2_seq = tf.reduce_mean(scene_2_min_compt_price_scene_2_seq, axis=1)
    scene_3_med_compt_price_scene_2_seq = tf.reduce_mean(scene_3_med_compt_price_scene_2_seq, axis=1)
    scene_3_min_compt_price_scene_2_seq = tf.reduce_mean(scene_3_min_compt_price_scene_2_seq, axis=1)

    scene_1_med_compt_price_seq = tf.reduce_sum(tf.multiply(scene_1_med_compt_price_seq, day_att_weight), axis=1)  # None*8
    scene_1_min_compt_price_seq = tf.reduce_sum(tf.multiply(scene_1_min_compt_price_seq, day_att_weight), axis=1)
    scene_2_med_compt_price_seq = tf.reduce_sum(tf.multiply(scene_2_med_compt_price_seq, day_att_weight), axis=1)
    scene_2_min_compt_price_seq = tf.reduce_sum(tf.multiply(scene_2_min_compt_price_seq, day_att_weight), axis=1)
    scene_3_med_compt_price_seq = tf.reduce_sum(tf.multiply(scene_3_med_compt_price_seq, day_att_weight), axis=1)
    scene_3_min_compt_price_seq = tf.reduce_sum(tf.multiply(scene_3_min_compt_price_seq, day_att_weight), axis=1)
    scene_1_simi_price_rate_seq = tf.reduce_sum(tf.multiply(scene_1_simi_price_rate_seq, simi_att_weight), axis=1)
    scene_2_simi_price_rate_seq = tf.reduce_sum(tf.multiply(scene_2_simi_price_rate_seq, simi_att_weight), axis=1)
    scene_3_simi_price_rate_seq = tf.reduce_sum(tf.multiply(scene_3_simi_price_rate_seq, simi_att_weight), axis=1)

    scene_1_compet_input = tf.concat(
        [scene_1_ord_price_rate_seq, scene_1_room_ord_price_rate_seq, scene_1_med_compt_price_scene_2_seq, scene_1_min_compt_price_scene_2_seq,
         scene_1_med_compt_price_seq, scene_1_min_compt_price_seq, scene_1_simi_price_rate_seq, scene_1_feats], axis=1)
    scene_2_compet_input = tf.concat(
        [scene_2_ord_price_rate_seq, scene_2_room_ord_price_rate_seq, scene_2_med_compt_price_scene_2_seq, scene_2_min_compt_price_scene_2_seq,
         scene_2_med_compt_price_seq, scene_2_min_compt_price_seq, scene_2_simi_price_rate_seq, scene_2_feats], axis=1)
    scene_3_compet_input = tf.concat(
        [scene_3_ord_price_rate_seq, scene_3_room_ord_price_rate_seq, scene_3_med_compt_price_scene_2_seq,
         scene_3_min_compt_price_scene_2_seq,
         scene_3_med_compt_price_seq, scene_3_min_compt_price_seq, scene_3_simi_price_rate_seq, scene_3_feats], axis=1)

    scene_1_compet_output = PCI(scene_1_compet_input, tf.estimator.ModeKeys.TRAIN, name='scene_1_compet',
                                                l2_reg=0.05,
                                                subexpert_nums=3)
    scene_2_compet_output = PCI(scene_2_compet_input, tf.estimator.ModeKeys.TRAIN, name='scene_2_compet',
                                               l2_reg=0.05,
                                               subexpert_nums=3)
    scene_3_compet_output = PCI(scene_3_compet_input, tf.estimator.ModeKeys.TRAIN, name='scene_3_compet',
                                              l2_reg=0.05,
                                              subexpert_nums=3)  ### None*64

    scene_1_input = tf.concat([scene_1_demand_output, scene_1_compet_output], axis=1)
    scene_2_input = tf.concat([scene_2_demand_output, scene_2_compet_output], axis=1)
    scene_3_input = tf.concat([scene_3_demand_output, scene_3_compet_output], axis=1)


    activation_fn = tf.nn.relu6
    scene_1_input = layers.batch_norm(scene_1_input, is_training=is_training, activation_fn=activation_fn,
                                    variables_collections=[dnn_parent_scope])
    scene_1_input1 = tf.keras.layers.Dense(128, name='scene_1_Dense1', use_bias=False)(scene_1_input)
    scene_1_input1 = layers.batch_norm(scene_1_input1, is_training=is_training, activation_fn=activation_fn,
                                      variables_collections=[dnn_parent_scope])
    scene_1_logit = tf.keras.layers.Dense(1, name='scene_1_Dense2', use_bias=True)(scene_1_input1)
    scene_1_logit = tf.sigmoid(scene_1_logit)

    scene_1_input2 = tf.keras.layers.Dense(128, name='scene_1_Dense3', use_bias=False)(scene_1_compet_input)  
    scene_1_input2 = layers.batch_norm(scene_1_input2, is_training=is_training, activation_fn=activation_fn,
                                     variables_collections=[dnn_parent_scope])
    scene_1_logit2 = tf.keras.layers.Dense(1, name='scene_1_Dense4', use_bias=True)(scene_1_input2)
    scene_1_logit2 = tf.sigmoid(scene_1_logit2) + 0.1

    scene_1_input3 = tf.keras.layers.Dense(128, name='scene_1_Dense5', use_bias=False)(scene_1_compet_input)
    scene_1_input3 = layers.batch_norm(scene_1_input3, is_training=is_training, activation_fn=activation_fn,
                                      variables_collections=[dnn_parent_scope])
    scene_1_logit3 = tf.keras.layers.Dense(1, name='scene_1_Dense6', use_bias=True)(scene_1_input3)
    scene_1_logit3 = tf.sigmoid(scene_1_logit3) + 0.3

    scene_2_input = layers.batch_norm(scene_2_input, is_training=is_training, activation_fn=activation_fn,
                                    variables_collections=[dnn_parent_scope])
    scene_2_input1 = tf.keras.layers.Dense(128, name='scene_2_Dense1', use_bias=False)(scene_2_input)
    scene_2_input1 = layers.batch_norm(scene_2_input1, is_training=is_training, activation_fn=activation_fn,
                                    variables_collections=[dnn_parent_scope])
    scene_2_logit = tf.keras.layers.Dense(1, name='scene_2_Dense2', use_bias=True)(scene_2_input1)
    scene_2_logit = tf.sigmoid(scene_2_logit)

    scene_2_input2 = tf.keras.layers.Dense(128, name='scene_2_Dense3', use_bias=False)(scene_2_compet_input)
    scene_2_input2 = layers.batch_norm(scene_2_input2, is_training=is_training, activation_fn=activation_fn,
                                     variables_collections=[dnn_parent_scope])
    scene_2_logit2 = tf.keras.layers.Dense(1, name='scene_2_Dense4', use_bias=True)(scene_2_input2)
    scene_2_logit2 = tf.sigmoid(scene_2_logit2) + 0.1

    scene_2_input3 = tf.keras.layers.Dense(128, name='scene_2_Dense5', use_bias=False)(scene_2_compet_input)
    scene_2_input3 = layers.batch_norm(scene_2_input3, is_training=is_training, activation_fn=activation_fn,
                                     variables_collections=[dnn_parent_scope])
    scene_2_logit3 = tf.keras.layers.Dense(1, name='scene_2_Dense6', use_bias=True)(scene_2_input3)
    scene_2_logit3 = tf.sigmoid(scene_2_logit3) + 0.3

    scene_3_input = layers.batch_norm(scene_3_input, is_training=is_training, activation_fn=activation_fn,
                                   variables_collections=[dnn_parent_scope])
    scene_3_input1 = tf.keras.layers.Dense(128, name='scene_3_Dense1', use_bias=False)(scene_3_input)
    scene_3_input1 = layers.batch_norm(scene_3_input1, is_training=is_training, activation_fn=activation_fn,
                                   variables_collections=[dnn_parent_scope])
    scene_3_logit = tf.keras.layers.Dense(1, name='scene_3_Dense2', use_bias=True)(scene_3_input1)
    scene_3_logit = tf.sigmoid(scene_3_logit)

    scene_3_input2 = tf.keras.layers.Dense(128, name='scene_3_Dense3', use_bias=False)(scene_3_compet_input)
    scene_3_input2 = layers.batch_norm(scene_3_input2, is_training=is_training, activation_fn=activation_fn,
                                    variables_collections=[dnn_parent_scope])
    scene_3_logit2 = tf.keras.layers.Dense(1, name='scene_3_Dense4', use_bias=True)(scene_3_input2)
    scene_3_logit2 = tf.sigmoid(scene_3_logit2) + 0.1

    scene_3_input3 = tf.keras.layers.Dense(128, name='scene_3_Dense5', use_bias=False)(scene_3_compet_input)
    scene_3_input3 = layers.batch_norm(scene_3_input3, is_training=is_training, activation_fn=activation_fn,
                                    variables_collections=[dnn_parent_scope])
    scene_3_logit3 = tf.keras.layers.Dense(1, name='scene_3_Dense6', use_bias=True)(scene_3_input3)
    scene_3_logit3 = tf.sigmoid(scene_3_logit3) + 0.3


    scene_1_label = tf.string_to_number(scene_1_label,out_type=tf.float32,name=None)
    scene_2_label = tf.string_to_number(scene_2_label,out_type=tf.float32,name=None)
    scene_3_label = tf.string_to_number(scene_3_label, out_type=tf.float32, name=None)
    checkin = tf.string_to_number(checkin, out_type=tf.int64,name=None)
    room_id = tf.string_to_number(room_id, out_type=tf.int64, name=None)

    scene_1_label = tf.reshape(scene_1_label, [-1, 1])
    scene_2_label = tf.reshape(scene_2_label, [-1, 1])
    scene_3_label = tf.reshape(scene_3_label, [-1, 1])
    checkin = tf.reshape(checkin, [-1, 1])
    room_id = tf.reshape(room_id, [-1, 1])

    tf.keras.backend.set_learning_phase(is_training)
    add_variables_from_scope('Dense',[ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.MODEL_VARIABLES])
    
    return room_id, checkin, scene_1_label, scene_2_label, scene_3_label, scene_1_logit, scene_1_logit2, scene_1_logit3, scene_2_logit, scene_2_logit2, scene_2_logit3, scene_3_logit, scene_3_logit2, scene_3_logit3, output_gate
