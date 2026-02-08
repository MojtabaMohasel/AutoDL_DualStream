import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, backend as K

@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class ConditionalPoolingLayer1D(layers.Layer):
    def __init__(self, pool_size, pool_type="max", **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.pool_type = pool_type.lower()
        if self.pool_type == "max":
            self.pooling_layer = layers.MaxPooling1D(pool_size=self.pool_size)
        elif self.pool_type == "average":
            self.pooling_layer = layers.AveragePooling1D(pool_size=self.pool_size)
        else:
            raise ValueError(f"Unsupported pool_type: {pool_type}. Choose 'max' or 'average'.")

    def call(self, inputs):
        input_shape = K.int_shape(inputs)
        if input_shape[1] is not None and input_shape[1] // self.pool_size > 0:
            return self.pooling_layer(inputs)
        else:
            raise ValueError(f"Negative dimension on 1D pooling.")


    def get_config(self):
        config = super().get_config()
        config.update({
            "pool_size": self.pool_size,
            "pool_type": self.pool_type,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class ConditionalPoolingLayer2D(layers.Layer):
    def __init__(self, pool_size=(2,2), pool_type="max", **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.pool_type = pool_type.lower()
        if self.pool_type == "max":
            self.pooling_layer = layers.MaxPooling2D(pool_size=self.pool_size)
        elif self.pool_type == "average":
            self.pooling_layer = layers.AveragePooling2D(pool_size=self.pool_size)
        else:
            raise ValueError(f"Unsupported pool_type: {pool_type}. Choose 'max' or 'average'.")

    def call(self, inputs):
        input_shape = K.int_shape(inputs)
        can_pool_h = input_shape[1] is None or (input_shape[1] // self.pool_size[0] > 0)
        can_pool_w = input_shape[2] is None or (input_shape[2] // self.pool_size[1] > 0)
        
        if can_pool_h and can_pool_w:
            return self.pooling_layer(inputs)
        else:
            raise ValueError(f"Negative dimension on 2D pooling.")


    def get_config(self):
        config = super().get_config()
        config.update({
            "pool_size": self.pool_size,
            "pool_type": self.pool_type,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



class ModelBuilder:
    def __init__(self, input_shape_1d, input_shape_2d, num_classes, config, logger):
        self.input_shape_1d = input_shape_1d
        self.input_shape_2d = input_shape_2d
        self.num_classes = num_classes
        self.config = config
        self.logger = logger
        self.active_streams = config.ga.ACTIVE_STREAMS



        self.CONV_GENE_ORDER = ['is_active', 'filters', 'kernel_size', 'activation', 'padding', 'strides', 'use_bn', 'use_pool', 'pool_type']
        self.POST_CONV_GENE_ORDER = ['global_pooling', 'use_dropout', 'dropout_rate']
        self.LSTM_GENE_ORDER = ['is_active', 'units', 'activation', 'recurrent_dropout', 'use_bn', 'use_dropout', 'dropout_rate']
        self.MLP_GENE_ORDER = ['is_active', 'use_bn', 'use_dense', 'units', 'activation', 'use_dropout', 'dropout_rate']
        self.NETWORK_PARAMS_GENE_ORDER = ['l1_reg', 'learning_rate']


        self.conv_gene_map = {name: i for i, name in enumerate(self.CONV_GENE_ORDER)}
        self.post_conv_gene_map = {name: i for i, name in enumerate(self.POST_CONV_GENE_ORDER)}
        self.lstm_gene_map = {name: i for i, name in enumerate(self.LSTM_GENE_ORDER)}
        self.mlp_gene_map = {name: i for i, name in enumerate(self.MLP_GENE_ORDER)}
        self.network_params_map = {name: i for i, name in enumerate(self.NETWORK_PARAMS_GENE_ORDER)}


        self.genes_per_conv_block = len(self.CONV_GENE_ORDER)
        self.genes_per_lstm_block = len(self.LSTM_GENE_ORDER)
        self.genes_per_mlp_block = len(self.MLP_GENE_ORDER)


        self.activation_map = dict(enumerate(self.config.mappings.ACTIVATION_MAP))
        self.padding_map = self.config.mappings.PADDING_MAP


        self.n_conv_blocks_1d = self.config.network_structure.N_CONV_BLOCKS_1D
        self.n_conv_blocks_2d = self.config.network_structure.N_CONV_BLOCKS_2D
        self.n_lstm_blocks = self.config.network_structure.N_LSTM_BLOCKS
        self.n_mlp_blocks = self.config.network_structure.N_MLP_BLOCKS



    def build_model(self, hp1d, hp2d, hp_lstm, hp_mlp, loss_gene, gamma, class_weights):
        """Builds a model dynamically based on the active streams in the config."""

        inputs = []
        feature_streams = []

        if not any(vars(self.active_streams).values()):
            raise ValueError("Model building error: At least one model stream must be active.")


        if self.active_streams.cnn_1d or self.active_streams.lstm:
            input_1d = layers.Input(shape=self.input_shape_1d, name="input_1d")
            inputs.append(input_1d)
            if self.active_streams.cnn_1d:
                cnn1d_features = self._build_1d_branch(input_1d, hp1d)
                feature_streams.append(cnn1d_features)
            if self.active_streams.lstm:
                lstm_features = self._build_lstm_branch(input_1d, hp_lstm)
                feature_streams.append(lstm_features)


        if self.active_streams.cnn_2d:
            input_2d = layers.Input(shape=self.input_shape_2d, name="input_2d")
            inputs.append(input_2d)
            cnn2d_features = self._build_2d_branch(input_2d, hp2d)
            feature_streams.append(cnn2d_features)


        if not feature_streams:
             raise ValueError("Model building error: No feature streams were built. Check active_streams config.")
        elif len(feature_streams) > 1:
            merged = layers.Concatenate(name="concatenated_features")(feature_streams)
        else:
            merged = feature_streams[0]

        output = self._build_mlp_head(merged, hp_mlp)
        model = models.Model(inputs=inputs, outputs=output, name="DynamicStreamIMUModel")


        num_network_params = len(self.NETWORK_PARAMS_GENE_ORDER)
        idx_network_params = len(hp_mlp) - num_network_params
        learning_rate = hp_mlp[idx_network_params + self.network_params_map['learning_rate']]
        l1_reg = hp_mlp[idx_network_params + self.network_params_map['l1_reg']]
        self._compile_model(model, loss_gene, gamma, class_weights, learning_rate, l1_reg)

        self._check_model_compiles(model)
        self.logger.info("Model built and compiled successfully.")
        return model



    def _compile_model(self, model, loss_gene, gamma_value, class_weights_values, learning_rate, l1_reg):
        if loss_gene == 0:
            loss_fn = keras.losses.CategoricalCrossentropy()
        else:
            loss_fn = keras.losses.CategoricalFocalCrossentropy(gamma=gamma_value, alpha=class_weights_values)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss_fn,
            metrics=["accuracy", tf.keras.metrics.F1Score(name="f1_score_macro", average="macro")]
        )

    def _check_model_compiles(self, model):
        try:
            if isinstance(model.input_shape, list):
                dummy_inputs = [tf.random.uniform((1, *shape[1:]), dtype=tf.float32) for shape in model.input_shape]
            else:

                dummy_inputs = tf.random.uniform((1, *model.input_shape[1:]), dtype=tf.float32)
            _ = model.predict(dummy_inputs, verbose=0)
            self.logger.debug("Model shape check passed with dummy data.")
        except Exception as e:
            self.logger.debug(f"Model shape check FAILED! {e}", exc_info=True)
            raise


    def _build_1d_branch(self, inputs, chromosome_1d):
        x = layers.BatchNormalization()(inputs)
        for block_idx in range(self.n_conv_blocks_1d):
            base_offset = block_idx * self.genes_per_conv_block
            if chromosome_1d[base_offset + self.conv_gene_map['is_active']]:
                try:

                    filters = int(chromosome_1d[base_offset + self.conv_gene_map['filters']])
                    kernel_size = int(chromosome_1d[base_offset + self.conv_gene_map['kernel_size']])
                    activation_idx = int(chromosome_1d[base_offset + self.conv_gene_map['activation']])
                    padding_idx = int(chromosome_1d[base_offset + self.conv_gene_map['padding']])
                    strides = int(chromosome_1d[base_offset + self.conv_gene_map['strides']])
                    use_bn = chromosome_1d[base_offset + self.conv_gene_map['use_bn']]
                    use_pool = chromosome_1d[base_offset + self.conv_gene_map['use_pool']]
                    pool_type_gene = chromosome_1d[base_offset + self.conv_gene_map['pool_type']]


                    activation = self.activation_map[activation_idx]

                    padding_keras = self.padding_map[padding_idx]
                    x = layers.Conv1D(filters=filters, kernel_size=kernel_size,
                                       activation=activation, padding=padding_keras,
                                       strides=strides)(x)
                    if use_bn:
                        x = layers.BatchNormalization()(x)
                    if use_pool:
                        pool_type = "max" if pool_type_gene == 0 else "average"
                        if K.int_shape(x)[1] is None or K.int_shape(x)[1] >= 2:
                            x = ConditionalPoolingLayer1D(pool_size=2, pool_type=pool_type)(x)
                        else:
                            self.logger.debug(f"Skipping 1D pooling in block {block_idx}")
                except Exception as e:
                    self.logger.debug(f"Error in 1D block {block_idx}: {e}")
                    raise

        idx_after_conv = self.n_conv_blocks_1d * self.genes_per_conv_block
        if chromosome_1d[idx_after_conv + self.post_conv_gene_map['global_pooling']]: x = layers.GlobalAveragePooling1D()(x)
        else: x = layers.Flatten()(x)
        if chromosome_1d[idx_after_conv + self.post_conv_gene_map['use_dropout']]:
            dropout_rate = chromosome_1d[idx_after_conv + self.post_conv_gene_map['dropout_rate']]
            x = layers.Dropout(dropout_rate)(x)
        if len(x.shape) > 2:
            x = layers.Flatten()(x)
        return x

    def _build_2d_branch(self, inputs, chromosome_2d):
        x = layers.BatchNormalization()(inputs)
        for block_idx in range(self.n_conv_blocks_2d):
            base_offset = block_idx * self.genes_per_conv_block
            if chromosome_2d[base_offset + self.conv_gene_map['is_active']]:
                try:

                    filters = int(chromosome_2d[base_offset + self.conv_gene_map['filters']])
                    kernel_size_val = int(chromosome_2d[base_offset + self.conv_gene_map['kernel_size']])
                    activation_idx = int(chromosome_2d[base_offset + self.conv_gene_map['activation']])
                    padding_idx = int(chromosome_2d[base_offset + self.conv_gene_map['padding']])
                    strides_val = int(chromosome_2d[base_offset + self.conv_gene_map['strides']])
                    use_bn = chromosome_2d[base_offset + self.conv_gene_map['use_bn']]
                    use_pool = chromosome_2d[base_offset + self.conv_gene_map['use_pool']]
                    pool_type_gene = chromosome_2d[base_offset + self.conv_gene_map['pool_type']]


                    activation = self.activation_map[(activation_idx)]

                    padding_keras = self.padding_map[padding_idx]

                    x = layers.Conv2D(filters=filters,
                                       kernel_size=(kernel_size_val, kernel_size_val),
                                       activation=activation,
                                       padding=padding_keras,
                                       strides=(strides_val, strides_val))(x)
                    if use_bn:
                        x = layers.BatchNormalization()(x)
                    if use_pool:
                        pool_type = "max" if pool_type_gene == 0 else "average"
                        shape = K.int_shape(x)
                        if (shape[1] or 0) >= 2 and (shape[2] or 0) >= 2:
                            x = ConditionalPoolingLayer2D(pool_size=(2,2), pool_type=pool_type)(x)
                        else:
                            self.logger.debug(f"Skipping 2D pooling in block {block_idx}")
                except Exception as e:
                    self.logger.debug(f"Error in 2D block {block_idx}: {e}")
                    raise

        idx_after_conv = self.n_conv_blocks_2d * self.genes_per_conv_block
        if chromosome_2d[idx_after_conv + self.post_conv_gene_map['global_pooling']]: x = layers.GlobalAveragePooling2D()(x)
        else: x = layers.Flatten()(x)
        if chromosome_2d[idx_after_conv + self.post_conv_gene_map['use_dropout']]:
            dropout_rate = chromosome_2d[idx_after_conv + self.post_conv_gene_map['dropout_rate']]
            x = layers.Dropout(dropout_rate)(x)
        if len(x.shape) > 2:
            x = layers.Flatten()(x)
        return x

    def _build_lstm_branch(self, inputs, chromosome_lstm):
        x = layers.BatchNormalization()(inputs)
        last_active_block_idx = -1
        for block_idx in range(self.n_lstm_blocks - 1, -1, -1):
            base_offset = block_idx * self.genes_per_lstm_block
            if chromosome_lstm[base_offset + self.lstm_gene_map['is_active']]:
                last_active_block_idx = block_idx
                break
        for block_idx in range(self.n_lstm_blocks):
            base_offset = block_idx * self.genes_per_lstm_block
            if chromosome_lstm[base_offset + self.lstm_gene_map['is_active']]:
                try:
                    units = int(chromosome_lstm[base_offset + self.lstm_gene_map['units']])
                    activation_idx = int(chromosome_lstm[base_offset + self.lstm_gene_map['activation']])
                    activation = self.activation_map[(activation_idx)]
                    recurrent_dropout = chromosome_lstm[base_offset + self.lstm_gene_map['recurrent_dropout']]
                    use_bn = chromosome_lstm[base_offset + self.lstm_gene_map['use_bn']]
                    use_dropout = chromosome_lstm[base_offset + self.lstm_gene_map['use_dropout']]
                    dropout_rate = chromosome_lstm[base_offset + self.lstm_gene_map['dropout_rate']]
                    return_sequences = (block_idx < last_active_block_idx)
                    x = layers.LSTM(units=units, activation=activation,
                                    recurrent_dropout=recurrent_dropout,
                                    return_sequences=return_sequences)(x)
                    if use_bn:
                        x = layers.BatchNormalization()(x)
                    if use_dropout:
                        x = layers.Dropout(dropout_rate)(x)
                except Exception as e:
                    self.logger.debug(f"Error in LSTM block {block_idx}: {e}")
                    raise
        if len(x.shape) > 2:
            x = layers.Flatten()(x)
        return x

    def _build_mlp_head(self, inputs, chromosome):
        x = inputs

        num_network_params = len(self.NETWORK_PARAMS_GENE_ORDER)
        idx_mlp_start = len(chromosome) - (self.n_mlp_blocks * self.genes_per_mlp_block) - num_network_params

        for block_idx in range(self.n_mlp_blocks):
            base_offset = idx_mlp_start + block_idx * self.genes_per_mlp_block
            if chromosome[base_offset + self.mlp_gene_map['is_active']]:
                try:
                    use_bn = chromosome[base_offset + self.mlp_gene_map['use_bn']]
                    use_dense = chromosome[base_offset + self.mlp_gene_map['use_dense']]
                    units = int(chromosome[base_offset + self.mlp_gene_map['units']])
                    activation_idx = int(chromosome[base_offset + self.mlp_gene_map['activation']])
                    use_dropout = chromosome[base_offset + self.mlp_gene_map['use_dropout']]
                    dropout_rate = chromosome[base_offset + self.mlp_gene_map['dropout_rate']]


                    if use_bn: x = layers.BatchNormalization()(x)
                    if use_dense:
                        activation = self.activation_map[(activation_idx)]
                        x = layers.Dense(units=units, activation=activation)(x)
                    if use_dropout: x = layers.Dropout(dropout_rate)(x)
                except Exception as e:
                    self.logger.debug(f"Error in MLP block {block_idx}: {e}")
                    raise

        idx_network_params = len(chromosome) - num_network_params
        l1_reg = chromosome[idx_network_params + self.network_params_map['l1_reg']]

        output = layers.Dense(self.num_classes, activation="softmax",
                              kernel_regularizer=keras.regularizers.l1(l1_reg))(x)
        return output


