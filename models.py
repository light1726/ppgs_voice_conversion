import tensorflow as tf


def get_biGRU_model(in_dim=346,
                    initial_hidden=256,
                    drop_rate=0.5,
                    gru_hidden=256,
                    out_dim=37):
    inputs = tf.keras.Input(shape=(None, in_dim,), name='ppg_lf0_inputs')
    seq_lengths = tf.keras.Input(shape=(), name='sequence_length')
    seq_mask = tf.sequence_mask(seq_lengths, name='sequence_bool_mask')
    dense_out = tf.keras.layers.Dense(
        initial_hidden, name='in_dense')(inputs)
    dense_out = tf.keras.layers.Dropout(drop_rate)(dense_out)
    dense_out = tf.keras.layers.ReLU(name='in_dense_relu')(dense_out)
    gru1_out = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            gru_hidden, return_sequences=True),
        input_shape=(None, initial_hidden),
        merge_mode='concat',
        name='biGRU_1')(dense_out, mask=seq_mask)
    gru2_out = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            gru_hidden, return_sequences=True),
        input_shape=(None, gru_hidden*2),
        merge_mode='concat',
        name='biGRU_2')(gru1_out, mask=seq_mask)
    gru3_out = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            gru_hidden, return_sequences=True),
        input_shape=(None, gru_hidden * 2),
        merge_mode='concat',
        name='biGRU_3')(gru2_out, mask=seq_mask)
    outputs = tf.keras.layers.Dense(
        out_dim, name='outputs')(gru3_out)
    model = tf.keras.Model(
        inputs=[inputs, seq_lengths], outputs=outputs)
    return model


if __name__ == '__main__':
    mdl = get_biGRU_model()
    mdl.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.losses.MeanSquaredError()])
    # inputs = np.random.random((100, 64, 346)).astype(np.float32)
    # targets = np.random.random((100, 64, 37)).astype(np.float32)
    # mdl.fit(x=inputs, y=targets, epochs=2, batch_size=4)
    mdl.summary()
