import tensorflow as tf

def PCI(input, mode, name, l2_reg = 0.05, subexpert_nums = 3, subexpert_units = '256,128'):
    """
    Price Competitiveness Integration
    """
    subexpert_units = list(map(int, subexpert_units.split(',')))
    subexperts = []
    for j in range(subexpert_nums):
        subexpert = input
        for i in range(len(subexpert_units)):
            subexpert = tf.layers.dense(inputs=subexpert, units=subexpert_units[i],
                                                  activation=tf.nn.relu,
                                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                  bias_initializer=tf.zeros_initializer(),
                                                  name='subexpert_%s_%d_%d' % (name, j, i))
            if mode == tf.estimator.ModeKeys.TRAIN:
                subexpert = tf.nn.dropout(subexpert, keep_prob=0.5)
        subexperts.append(subexpert)
    subexperts = tf.concat([tf.expand_dims(se, axis=1) for se in subexperts], axis=1) # None * subexpert_nums * 64
    gate_network = tf.contrib.layers.fully_connected(
        inputs=input,
        num_outputs=subexpert_nums,
        activation_fn=tf.nn.relu,
        weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
    gate_network_shape = gate_network.get_shape().as_list()
    gate_network = tf.expand_dims(tf.nn.softmax(gate_network), axis=2)
    gate_network = tf.reshape(gate_network, shape=[-1, gate_network_shape[1], 1])  # None * subexpert_nums * 1
    output = tf.multiply(subexperts, gate_network) # None * subexpert_nums * 64
    output = tf.reduce_sum(output, axis=1) # None * 64
    return output

def PLE_emb(input, task_num, expert_num_per_task, expert_num_shared, expert_size, level_number):
    ple_input = input
    for i in range(0, level_number):
        if i == level_number - 1:
            return SinglePLE(ple_input, 'lev_' + str(i), task_num, expert_num_per_task, expert_num_shared, expert_size, True)
        else:
            outputs,_ = SinglePLE(ple_input, 'lev_' + str(i), task_num, expert_num_per_task, expert_num_shared, expert_size, False)
            ple_input = outputs


#### input:[task1,task2,...concat(task1,task2,...)]
def SinglePLE(input, name, task_num, expert_num_per_task, expert_num_shared, expert_size, if_last):
    # task-specific expert part
    expert_outputs = []
    for i in range(0, task_num):
        subexpert = input[i]
        for j in range(0, expert_num_per_task):
            subexpert = tf.layers.dense(inputs=subexpert, units=expert_size,
                                        activation=tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        bias_initializer=tf.zeros_initializer(),
                                        name='task-expert_%s_%d_%d' % (name, j, i))
            expert_outputs.append(subexpert)

    # shared expert part
    for i in range(0, expert_num_shared):
        subexpert = tf.layers.dense(inputs=input[-1], units=expert_size,
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.zeros_initializer(),
                                    name='share-expert_%s_%d' % (name, i))
        expert_outputs.append(subexpert)

    # task gate part
    task_gate_list = []
    output_gate_list = []
    task_gate_num = expert_num_per_task + expert_num_shared
    for i in range(0, task_num):
        gate_network = tf.contrib.layers.fully_connected(
            inputs=input[i],
            num_outputs=task_gate_num,
            activation_fn=tf.nn.relu,
            weights_regularizer=tf.contrib.layers.l2_regularizer(0.05))
        gate_network = tf.nn.softmax(gate_network)
        output_gate_list.append(gate_network)
        gate_network = tf.expand_dims(gate_network, axis=2)   # None * task_gate_num * 1
        task_gate_list.append(gate_network)
    output_gate = tf.concat([id for id in output_gate_list], axis=1)
    print('output_gate shape:', output_gate.get_shape())

    # attention compute
    outputs = []
    for i in range(0, task_num):
        cur_experts = expert_outputs[i * expert_num_per_task:(i + 1) * expert_num_per_task] + expert_outputs[
                                                                                              -int(expert_num_shared):]
        expert_concat = tf.concat([tf.expand_dims(id, axis=1) for id in cur_experts], axis=1)  # None*task_gate_num*expert_size

        expert_att = tf.reduce_sum(tf.multiply(expert_concat, task_gate_list[i]), axis=1)  # None*expert_size
        outputs.append(expert_att)

    # shared gate part
    if not if_last:
        shared_gate_num = task_num * expert_num_per_task + expert_num_shared
        gate_network = tf.contrib.layers.fully_connected(
            inputs=input[-1],
            num_outputs=shared_gate_num,
            activation_fn=tf.nn.relu,
            weights_regularizer=tf.contrib.layers.l2_regularizer(0.05))
        share_gate = tf.expand_dims(tf.nn.softmax(gate_network), axis=2)  # None * shared_gate_num * 1
        expert_concat = tf.concat([tf.expand_dims(id, axis=1) for id in expert_outputs], axis=1)  # None*shared_gate_num*expert_size
        expert_att = tf.reduce_sum(tf.multiply(expert_concat, share_gate), axis=1)  # None*expert_size
        outputs.append(expert_att)

    return outputs,output_gate