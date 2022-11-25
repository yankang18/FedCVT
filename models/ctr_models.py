import torch
import torch.nn as nn

from models.ctr_models_utils import build_input_features, SparseFeat, DenseFeat, VarLenSparseFeat, \
    create_embedding_matrix, combined_dnn_input, DNN


class BaseModel(nn.Module):
    def __init__(self, linear_feature_columns, dnn_feature_columns, l2_reg_linear=1e-5, l2_reg_embedding=1e-5,
                 init_std=0.0001, seed=1024, task='binary', device='cpu', gpus=None):

        super(BaseModel, self).__init__()
        torch.manual_seed(seed)
        self.dnn_feature_columns = dnn_feature_columns

        self.device = device
        self.gpus = gpus
        if gpus and str(self.gpus[0]) not in self.device:
            raise ValueError(
                "`gpus[0]` should be the same gpu with `device`")
        print(linear_feature_columns)
        print(dnn_feature_columns)
        self.feature_index = build_input_features(
            linear_feature_columns + dnn_feature_columns)
        self.dnn_feature_columns = dnn_feature_columns

        self.embedding_dict = create_embedding_matrix(dnn_feature_columns, init_std, sparse=False, device=device)
        # nn.ModuleDict(
        #     {feat.embedding_name: nn.Embedding(feat.dimension, embedding_size, sparse=True) for feat in
        #      self.dnn_feature_columns}
        # )

        self.regularization_weight = []

        self.add_regularization_weight(self.embedding_dict.parameters(), l2=l2_reg_embedding)

        # self.out = PredictionLayer(task, )
        self.to(device)

    def input_from_feature_columns(self, X, feature_columns, embedding_dict, support_dense=True):

        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        # varlen_sparse_feature_columns = list(
        #     filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError(
                "DenseFeat is not supported in dnn_feature_columns")

        sparse_embedding_list = [embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in sparse_feature_columns]

        # sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
        #                                               varlen_sparse_feature_columns)
        # varlen_sparse_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
        #                                                        varlen_sparse_feature_columns, self.device)

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            dense_feature_columns]

        return sparse_embedding_list, dense_value_list

    def compute_input_dim(self, feature_columns, include_sparse=True, include_dense=True, feature_group=False):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        dense_input_dim = sum(
            map(lambda x: x.dimension, dense_feature_columns))
        if feature_group:
            sparse_input_dim = len(sparse_feature_columns)
        else:
            sparse_input_dim = sum(feat.embedding_dim for feat in sparse_feature_columns)
        input_dim = 0
        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim
        return input_dim

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    @property
    def embedding_size(self, ):
        feature_columns = self.dnn_feature_columns
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        embedding_size_set = set([feat.embedding_dim for feat in sparse_feature_columns])
        if len(embedding_size_set) > 1:
            raise ValueError("embedding_dim of SparseFeat and VarlenSparseFeat must be same in this model!")
        return list(embedding_size_set)[0]


class DNNFM(BaseModel):
    """Instantiates the DeepFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param use_fm: bool,use FM part or not
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self, an_id,
                 linear_feature_columns, dnn_feature_columns, use_fm=False,
                 dnn_hidden_units=(256, 64),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0.,
                 dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None):

        super(DNNFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                    l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                    device=device, gpus=gpus)
        self.id = an_id
        self.dnn_feature_columns = dnn_feature_columns

        self.reg_loss = torch.zeros((1,), device=device)
        self.aux_loss = torch.zeros((1,), device=device)
        self.device = device
        self.gpus = gpus
        if gpus and str(self.gpus[0]) not in self.device:
            raise ValueError(
                "`gpus[0]` should be the same gpu with `device`")

        self.feature_index = build_input_features(
            linear_feature_columns + dnn_feature_columns)
        self.dnn_feature_columns = dnn_feature_columns

        self.embedding_dict = create_embedding_matrix(dnn_feature_columns, init_std, sparse=False, device=device)

        self.add_regularization_weight(self.embedding_dict.parameters(), l2=l2_reg_embedding)

        self.use_fm = use_fm
        self.use_dnn = len(dnn_feature_columns) > 0 and len(
            dnn_hidden_units) > 0

        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)

            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
        self.to(device)

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        if self.use_dnn:
            dnn_input = combined_dnn_input(
                sparse_embedding_list, dense_value_list)

            dnn_output = self.dnn(dnn_input)

        return dnn_output


class Converter_DNNFM(BaseModel):
    """Instantiates the DeepFM Network architecture.
    """

    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, use_fm=False,
                 dnn_hidden_units=[256],
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0.,
                 dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None):

        super(Converter_DNNFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                              l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed,
                                              task=task,
                                              device=device, gpus=gpus)

        self.use_fm = use_fm
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)

            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
        self.to(device)

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)

        if self.use_dnn:
            dnn_input = combined_dnn_input(
                sparse_embedding_list, dense_value_list)

            dnn_output = self.dnn(dnn_input)

        return dnn_output


class Backbone_DNNFM(nn.Module):
    def __init__(self, hidden_dims=[256, 128], use_BN=False, dropout_rate=0.0):
        super(Backbone_DNNFM, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        if use_BN:
            self.layer2 = nn.Sequential(
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.BatchNorm1d(hidden_dims[1]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            )
        else:
            self.layer2 = nn.Sequential(
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                # nn.BatchNorm1d(hidden_dims[1]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            )

    def forward(self, x):
        x = self.layer2(x)
        return x


def get_dnnfm(struct_config):
    struct_config = {} if struct_config is None else struct_config
    col_names = struct_config.get('col_names')

    model = DNNFM(col_names, col_names, dnn_activation="leakyrelu", init_std=0.0001)
    return model
