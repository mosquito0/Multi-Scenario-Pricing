# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import layers
import copy

# -----------------------------------------------------------------------------------------------------------
# fg parse class
FEATURE_KEY = 'features'
FEATURE_NAME_KEY = 'feature_name'
SEQUENCE_FEATURE_NAME_KEY = 'sequence_name'
VALUE_TYPE_KEY = 'value_type'
SHARE_KEY = 'shared'
SHARE_MATRIX_NAME_KEY = 'shared_matrix_name'
LR_KEY = 'is_wide'
NORMALIZER = 'normalizer'

linear_parent_scope = "linear"
dnn_parent_scope = "dnn"

initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)


class FeatureColumnGenerator(object):
    def __init__(self, feature_configs):
        self.features = {}
        self.seq_features = {}
        self.shared_features = {}
        # sharing a embed matrix and store as a list
        self.wide_features = {}

        self._parse(feature_configs)

    def _parse(self, feature_configs):
        features = feature_configs[FEATURE_KEY]
        for feature in features:
            if LR_KEY in feature:
                column = self.build_wide_feature(feature)
                if column: self.wide_features[feature[FEATURE_NAME_KEY]] = column
            elif FEATURE_NAME_KEY in feature:
                column = self.build_feature(feature)
                if column: self.features[feature[FEATURE_NAME_KEY]] = column
            elif SEQUENCE_FEATURE_NAME_KEY in feature:
                column = self.build_seq_feature(feature)
                if column: self.seq_features[feature[SEQUENCE_FEATURE_NAME_KEY]] = column
            else:
                tf.logging.info("conf error")
                raise ValueError

    def build_feature(self, feature):
        feature_name = feature[FEATURE_NAME_KEY]
        value_type = feature[VALUE_TYPE_KEY]
        tf.logging.info("building feature: {}".format(feature_name))
        if value_type == "string":
            id_feature = layers.sparse_column_with_hash_bucket(
                column_name=feature_name,
                hash_bucket_size=feature['hash_bucket_size']
            )

            if SHARE_MATRIX_NAME_KEY in feature:
                shared_matrix_name = feature[SHARE_MATRIX_NAME_KEY]
                if shared_matrix_name in self.shared_features:
                    # seq_no = -1 means it is not seq type
                    self.shared_features[shared_matrix_name].append(
                        (feature_name, id_feature, -1, feature['embed_size'], feature_name))
                else:
                    self.shared_features[shared_matrix_name] = [
                        (feature_name, id_feature, -1, feature['embed_size'], feature_name)]
                return None
            else:
                return layers.embedding_column(
                    id_feature,
                    dimension=feature['embed_size'],
                    initializer=initializer
                )
        elif value_type == 'double':
            normalizer = None
            if NORMALIZER in feature:
                params = feature[NORMALIZER].split(',')
                type = params[0]
                p1 = float(params[1])
                p2 = float(params[2])
                if type == 'minmax':
                    normalizer = self.get_min_max_normalizer(p1, p2)
                elif type == 'zscore':
                    normalizer = self.get_zscore_normalizer(p1, p2)
                else:
                    tf.logging.info("conf error")
                    raise ValueError
            return layers.real_valued_column(
                column_name=feature_name,
                default_value=0.0,
                dimension=feature['value_dimension'] if 'value_dimension' in feature else 1,
                normalizer=normalizer
            )

    def build_seq_feature(self, feature):
        sequence_name = feature[SEQUENCE_FEATURE_NAME_KEY]
        flatten_feature_configs = self.flatten_seq_feature(feature)
        field_dict = {}
        for config in flatten_feature_configs:
            # for every (step, att)
            feature_name = config["sequence_feature_name"]
            value_type = config[VALUE_TYPE_KEY]
            sequence_no = config["sequence_no"]
            feature_name_field = config["feature_name"]
            tf.logging.info("building feature: {}".format(feature_name))
            if value_type == "string":
                id_feature = layers.sparse_column_with_hash_bucket(
                    column_name=feature_name,
                    hash_bucket_size=config['hash_bucket_size']
                )
                if SHARE_MATRIX_NAME_KEY in config:
                    shared_matrix_name = config[SHARE_MATRIX_NAME_KEY]
                    if shared_matrix_name in self.shared_features:
                        # seq_no = -1 means it is not seq type
                        self.shared_features[shared_matrix_name].append(
                            (feature_name_field, id_feature, sequence_no, config['embed_size'], feature_name))
                    else:
                        self.shared_features[shared_matrix_name] = [
                            (feature_name_field, id_feature, sequence_no, config['embed_size'], feature_name)]
                    continue
                else:
                    column = layers.embedding_column(
                        id_feature,
                        dimension=config['embed_size'],
                        initializer=initializer
                    )
            elif value_type == 'double':
                column = layers.real_valued_column(
                    column_name=feature_name,
                    default_value=0.0,
                    dimension=config['value_dimension'] if 'value_dimension' in feature else 1
                )
            else:
                tf.logging.info("value type error: {}".format(value_type))
                raise ValueError

            if feature_name_field in field_dict:
                field_dict[feature_name_field].append((column, feature_name, sequence_no))
            else:
                field_dict[feature_name_field] = [(column, feature_name, sequence_no)]

        for field in field_dict:
            field_dict[field] = sorted(field_dict[field], key=lambda x: x[2])

        return field_dict

    def flatten_seq_feature(self, seq_feature_config):
        seq_num = seq_feature_config["sequence_length"]
        seq_name = seq_feature_config["sequence_name"]
        seq_config = copy.copy(seq_feature_config)
        del seq_config['features']
        flatten_feature_configs = []
        for feature_config_tmpl in seq_feature_config["features"]:
            for seq_no in xrange(0, seq_num):
                seq_feature_config = copy.deepcopy(feature_config_tmpl)
                seq_feature_config["sequence_no"] = seq_no
                seq_feature_config["sequence_feature_name"] = (
                        "%s_%d_%s" % (seq_name, seq_no, seq_feature_config["feature_name"]))
                seq_feature_config['sequence_config'] = seq_config
                flatten_feature_configs.append(seq_feature_config)
        return flatten_feature_configs

    def build_wide_feature(self, feature):
        feature_name = feature[FEATURE_NAME_KEY]
        value_type = feature[VALUE_TYPE_KEY]
        if value_type == "string":
            id_feature = layers.sparse_column_with_hash_bucket(
                column_name=feature_name,
                hash_bucket_size=feature['hash_bucket_size']
            )
            return id_feature
        else:
            tf.logging.info("wide must be id type")
            raise ValueError

    def get_columns(self):
        return self.features

    def get_sequence_columns(self):
        return self.seq_features

    def get_wide_columns(self, black_list):
        # in wide, drop feats in black_list
        black_list_feats = black_list.split(',')
        tf.logging.info("black_list: {}".format(black_list_feats))
        names = []
        columns = []
        for name in self.wide_features:
            if name in black_list_feats:
                continue
            names.append(name)
            columns.append(self.wide_features[name])
        # wide logit is too large, for a small initialized value
        wide_columns = layers.shared_embedding_columns(columns, dimension=1,
                                                       shared_embedding_name='wide_matrix',
                                                       initializer=initializer)
        return wide_columns

    def get_shared_columns(self):
        shared_features_ = {}
        shared_seq_features_ = {}
        for shared_matrix_name in self.shared_features:
            coms = self.shared_features[shared_matrix_name]
            feature_field_names, id_features, sequence_nos, embed_sizes, feature_names = zip(*coms)
            if len(set(embed_sizes)) != 1:
                tf.logging.info("shared error {}".format(shared_matrix_name))
                raise ValueError
            columns = layers.shared_embedding_columns(
                id_features,
                dimension=embed_sizes[0],
                shared_embedding_name=shared_matrix_name,
                initializer=initializer
            )
            coms = zip(feature_field_names, columns, sequence_nos, feature_names)
            for com in coms:
                feature_field_name, column, sequence_no, feature_name = com
                if sequence_no == -1:
                    # no seq type
                    shared_features_[feature_field_name] = column
                else:
                    if feature_field_name in shared_seq_features_:
                        shared_seq_features_[feature_field_name].append((column, feature_name, sequence_no))
                    else:
                        shared_seq_features_[feature_field_name] = [(column, feature_name, sequence_no)]
        for field in shared_seq_features_:
            shared_seq_features_[field] = sorted(shared_seq_features_[field], key=lambda x: x[2])
        return shared_features_, shared_seq_features_

    def get_output_dict(self, features):
        """
        :param features: fg result_dict
        :return: output dict
        """
        normal_columns = self.features
        sequence_columns = self.seq_features
        shared_normal_columns, shared_sequence_columns = self.get_shared_columns()
        normal_columns.update(shared_normal_columns)
        sequence_columns.update(shared_sequence_columns)
        outputs_dict = {}
        tf.logging.info("building normal feature column...")
        for f_name in normal_columns:
            input_feature = {f_name: features[f_name]}
            input_column = [normal_columns[f_name]]
            tf.logging.info("building feature column: {}".format(f_name))
            outputs_dict[f_name] = layers.input_from_feature_columns(input_feature, input_column,
                                                                     weight_collections=[dnn_parent_scope])

        tf.logging.info("building sequence feature column...")
        for seq_name in sequence_columns:
            field_dict = sequence_columns[seq_name]
            if isinstance(field_dict, dict):
                for field in field_dict:
                    pairs = field_dict[field]
                    output_list = []
                    for pair in pairs:
                        column, feature_name, sequence_no = pair
                        input_feature = {feature_name: features[feature_name]}
                        input_column = [column]
                        tf.logging.info("building feature column: {}".format(feature_name))
                        output_list.append(layers.input_from_feature_columns(input_feature, input_column,
                                                                             weight_collections=[dnn_parent_scope]))
                    outputs_dict[seq_name + '_' + field] = output_list
            elif isinstance(field_dict, list):
                pairs = field_dict
                output_list = []
                for pair in pairs:
                    column, feature_name, sequence_no = pair
                    input_feature = {feature_name: features[feature_name]}
                    input_column = [column]
                    tf.logging.info("building feature column: {}".format(feature_name))
                    output_list.append(layers.input_from_feature_columns(input_feature, input_column,
                                                                         weight_collections=[dnn_parent_scope]))
                outputs_dict['shared' + '_' + seq_name] = output_list
        return outputs_dict

    def get_min_max_normalizer(self, min, max):
        return lambda x: (x - min) / (max - min)

    def get_zscore_normalizer(self, mean, std):
        return lambda x: (x - mean) / std
