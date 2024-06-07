import textwrap
from collections import OrderedDict
from copy import deepcopy

from torch.distributions import Normal
from torch.func import jacrev, functional_call

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as D
import torch.nn as nn

from diffusion_policy.configs import DiffusionModelRunConfig


class ConditionEncoder(nn.Module):
    def __init__(self, cond_shape_dict):
        super().__init__()
        self.cond_encoders = nn.ModuleDict()
        self.cond_dim = 0
        for name, shape in cond_shape_dict.items():
            if len(shape) == 1:
                self.cond_encoders[name] = nn.Identity()
                self.cond_dim += shape[0]
            else:
                raise NotImplementedError("Unsupported condition shape")

    def forward(self, cond_dict):
        cond = [self.cond_encoders[k](v) for k, v in sorted(cond_dict.items())]
        return torch.cat(cond, dim=-1)


class TanhWrappedDistribution(D.Distribution):
    """
    Class that wraps another valid torch distribution, such that sampled values from the base distribution are
    passed through a tanh layer. The corresponding (log) probabilities are also modified accordingly.
    Tanh Normal distribution - adapted from rlkit and CQL codebase
    (https://github.com/aviralkumar2907/CQL/blob/d67dbe9cf5d2b96e3b462b6146f249b3d6569796/d4rl/rlkit/torch/distributions.py#L6).
    """

    def __init__(self, base_dist, scale=1.0, epsilon=1e-6):
        """
        Args:
            base_dist (Distribution): Distribution to wrap with tanh output
            scale (float): Scale of output
            epsilon (float): Numerical stability epsilon when computing log-prob.
        """
        self.base_dist = base_dist
        self.scale = scale
        self.tanh_epsilon = epsilon
        super(TanhWrappedDistribution, self).__init__()

    def log_prob(self, value, pre_tanh_value=None):
        """
        Args:
            value (torch.Tensor): some tensor to compute log probabilities for
            pre_tanh_value: If specified, will not calculate atanh manually from @value. More numerically stable
        """
        value = value / self.scale
        if pre_tanh_value is None:
            one_plus_x = (1. + value).clamp(min=self.tanh_epsilon)
            one_minus_x = (1. - value).clamp(min=self.tanh_epsilon)
            pre_tanh_value = 0.5 * torch.log(one_plus_x / one_minus_x)
        lp = self.base_dist.log_prob(pre_tanh_value)
        tanh_lp = torch.log(1 - value * value + self.tanh_epsilon)
        # In case the base dist already sums up the log probs, make sure we do the same
        return lp - tanh_lp if len(lp.shape) == len(tanh_lp.shape) else lp - tanh_lp.sum(-1)

    def sample(self, sample_shape=torch.Size(), return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.
        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.base_dist.sample(sample_shape=sample_shape).detach()

        if return_pretanh_value:
            return torch.tanh(z) * self.scale, z
        else:
            return torch.tanh(z) * self.scale

    def rsample(self, sample_shape=torch.Size(), return_pretanh_value=False):
        """
        Sampling in the reparameterization case - for differentiable samples.
        """
        z = self.base_dist.rsample(sample_shape=sample_shape)

        if return_pretanh_value:
            return torch.tanh(z) * self.scale, z
        else:
            return torch.tanh(z) * self.scale

    @property
    def mean(self):
        return self.base_dist.mean

    @property
    def stddev(self):
        return self.base_dist.stddev


class MLP(nn.Module):
    """
    Base class for simple Multi-Layer Perceptrons.
    """

    def __init__(
            self,
            input_dim,
            output_dim,
            layer_dims=(),
            layer_func=nn.Linear,
            layer_func_kwargs=None,
            activation=nn.ReLU,
            dropouts=None,
            normalization=True,
            output_activation=None,
    ):
        """
        Args:
            input_dim (int): dimension of inputs

            output_dim (int): dimension of outputs

            layer_dims ([int]): sequence of integers for the hidden layers sizes

            layer_func: mapping per layer - defaults to Linear

            layer_func_kwargs (dict): kwargs for @layer_func

            activation: non-linearity per layer - defaults to ReLU

            dropouts ([float]): if not None, adds dropout layers with the corresponding probabilities
                after every layer. Must be same size as @layer_dims.

            normalization (bool): if True, apply layer normalization after each layer

            output_activation: if provided, applies the provided non-linearity to the output layer
        """
        super(MLP, self).__init__()
        layers = []
        dim = input_dim
        if layer_func_kwargs is None:
            layer_func_kwargs = dict()
        if dropouts is not None:
            assert (len(dropouts) == len(layer_dims))
        for i, l in enumerate(layer_dims):
            layers.append(layer_func(dim, l, **layer_func_kwargs))
            if normalization:
                layers.append(nn.LayerNorm(l))
            layers.append(activation())
            if dropouts is not None and dropouts[i] > 0.:
                layers.append(nn.Dropout(dropouts[i]))
            dim = l
        layers.append(layer_func(dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation())
        self._layer_func = layer_func
        self.nets = layers
        self._model = nn.Sequential(*layers)

        self._layer_dims = layer_dims
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._dropouts = dropouts
        self._act = activation
        self._output_act = output_activation

    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        return [self._output_dim]

    def forward(self, inputs):
        """
        Forward pass.
        """
        return self._model(inputs)

    def __repr__(self):
        """Pretty print network."""
        header = str(self.__class__.__name__)
        act = None if self._act is None else self._act.__name__
        output_act = None if self._output_act is None else self._output_act.__name__

        indent = ' ' * 4
        msg = "input_dim={}\noutput_dim={}\nlayer_dims={}\nlayer_func={}\ndropout={}\nact={}\noutput_act={}".format(
            self._input_dim, self._output_dim, self._layer_dims,
            self._layer_func.__name__, self._dropouts, act, output_act
        )
        msg = textwrap.indent(msg, indent)
        msg = header + '(\n' + msg + '\n)'
        return msg


class ObservationDecoder(nn.Module):
    """
    Module that can generate observation outputs by modality. Inputs are assumed
    to be flat (usually outputs from some hidden layer). Each observation output
    is generated with a linear layer from these flat inputs. Subclass this
    module in order to implement more complex schemes for generating each
    modality.
    """

    def __init__(
            self,
            decode_shapes,
            input_feat_dim,
    ):
        """
        Args:
            decode_shapes (OrderedDict): a dictionary that maps observation key to
                expected shape. This is used to generate output modalities from the
                input features.

            input_feat_dim (int): flat input dimension size
        """
        super(ObservationDecoder, self).__init__()

        # important: sort observation keys to ensure consistent ordering of modalities
        assert isinstance(decode_shapes, OrderedDict)
        self.obs_shapes = OrderedDict()
        for k in decode_shapes:
            self.obs_shapes[k] = decode_shapes[k]

        self.input_feat_dim = input_feat_dim
        self._create_layers()

    def _create_layers(self):
        """
        Create a linear layer to predict each modality.
        """
        self.nets = nn.ModuleDict()
        for k in self.obs_shapes:
            layer_out_dim = int(np.prod(self.obs_shapes[k]))
            self.nets[k] = nn.Linear(self.input_feat_dim, layer_out_dim)

    def output_shape(self, input_shape=None):
        """
        Returns output shape for this module, which is a dictionary instead
        of a list since outputs are dictionaries.
        """
        return {k: list(self.obs_shapes[k]) for k in self.obs_shapes}

    def forward(self, feats):
        """
        Predict each modality from input features, and reshape to each modality's shape.
        """
        output = {}
        for k in self.obs_shapes:
            out = self.nets[k](feats)
            output[k] = out.reshape(-1, *self.obs_shapes[k])
        return output

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        for k in self.obs_shapes:
            msg += textwrap.indent('\nKey(\n', ' ' * 4)
            indent = ' ' * 8
            msg += textwrap.indent("name={}\nshape={}\n".format(k, self.obs_shapes[k]), indent)
            msg += textwrap.indent("net=({})\n".format(self.nets[k]), indent)
            msg += textwrap.indent(")", ' ' * 4)
        msg = header + '(' + msg + '\n)'
        return msg


class MIMO_MLP(nn.Module):
    """
    Extension to MLP to accept multiple observation dictionaries as input and
    to output dictionaries of tensors. Inputs are specified as a dictionary of
    observation dictionaries, with each key corresponding to an observation group.

    This module utilizes @ObservationGroupEncoder to process the multiple input dictionaries and
    @ObservationDecoder to generate tensor dictionaries. The default behavior
    for encoding the inputs is to process visual inputs with a learned CNN and concatenating
    the flat encodings with the other flat inputs. The default behavior for generating
    outputs is to use a linear layer branch to produce each modality separately
    (including visual outputs).
    """

    def __init__(
            self,
            cond_shape_dict,
            output_shapes,
            layer_dims,
            layer_func=nn.Linear,
            activation=nn.ReLU,
            encoder_kwargs=None,
    ):
        """
        Args:
            input_obs_group_shapes (OrderedDict): a dictionary of dictionaries.
                Each key in this dictionary should specify an observation group, and
                the value should be an OrderedDict that maps modalities to
                expected shapes.

            output_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for outputs.

            layer_dims ([int]): sequence of integers for the MLP hidden layer sizes

            layer_func: mapping per MLP layer - defaults to Linear

            activation: non-linearity per MLP layer - defaults to ReLU

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """
        super(MIMO_MLP, self).__init__()

        assert isinstance(cond_shape_dict, OrderedDict)
        assert isinstance(output_shapes, OrderedDict)

        self.input_obs_group_shapes = cond_shape_dict
        self.output_shapes = output_shapes

        self.nets = nn.ModuleDict()

        # Encoder for all observation groups.
        self.nets["encoder"] = ConditionEncoder(cond_shape_dict)

        # flat encoder output dimension
        mlp_input_dim = self.nets["encoder"].cond_dim

        # intermediate MLP layers
        self.nets["mlp"] = MLP(
            input_dim=mlp_input_dim,
            output_dim=layer_dims[-1],
            layer_dims=layer_dims[:-1],
            layer_func=layer_func,
            activation=activation,
            output_activation=activation,  # make sure non-linearity is applied before decoder
        )

        # decoder for output modalities
        self.nets["decoder"] = ObservationDecoder(
            decode_shapes=self.output_shapes,
            input_feat_dim=layer_dims[-1],
        )

    def output_shape(self, input_shape=None):
        """
        Returns output shape for this module, which is a dictionary instead
        of a list since outputs are dictionaries.
        """
        return {k: list(self.output_shapes[k]) for k in self.output_shapes}

    def forward(self, inputs):
        """
        Process each set of inputs in its own observation group.

        Args:
            inputs (dict): a dictionary of dictionaries with one dictionary per
                observation group. Each observation group's dictionary should map
                modality to torch.Tensor batches. Should be consistent with
                @self.input_obs_group_shapes.

        Returns:
            outputs (dict): dictionary of output torch.Tensors, that corresponds
                to @self.output_shapes
        """
        enc_outputs = self.nets["encoder"](inputs)
        mlp_out = self.nets["mlp"](enc_outputs)
        return self.nets["decoder"](mlp_out)

    def _to_string(self):
        """
        Subclasses should override this method to print out info about network / policy.
        """
        return ''

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 4
        if self._to_string() != '':
            msg += textwrap.indent("\n" + self._to_string() + "\n", indent)
        msg += textwrap.indent("\nencoder={}".format(self.nets["encoder"]), indent)
        msg += textwrap.indent("\n\nmlp={}".format(self.nets["mlp"]), indent)
        msg += textwrap.indent("\n\ndecoder={}".format(self.nets["decoder"]), indent)
        msg = header + '(' + msg + '\n)'
        return msg


class ActorNetwork(MIMO_MLP):
    """
    A basic policy network that predicts actions from observations.
    Can optionally be goal conditioned on future observations.
    """

    def __init__(
            self,
            obs_shapes,
            ac_dim,
            mlp_layer_dims,
            goal_shapes=None,
            encoder_kwargs=None,
    ):
        """
        Args:
            obs_shapes (OrderedDict): a dictionary that maps observation keys to
                expected shapes for observations.

            ac_dim (int): dimension of action space.

            mlp_layer_dims ([int]): sequence of integers for the MLP hidden layers sizes.

            goal_shapes (OrderedDict): a dictionary that maps observation keys to
                expected shapes for goal observations.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-observation key information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """
        assert isinstance(obs_shapes, OrderedDict)
        self.obs_shapes = obs_shapes
        self.ac_dim = ac_dim

        # set up different observation groups for @MIMO_MLP

        self._is_goal_conditioned = False

        self.goal_shapes = OrderedDict()

        output_shapes = self._get_output_shapes()
        super(ActorNetwork, self).__init__(
            cond_shape_dict=self.obs_shapes,
            output_shapes=output_shapes,
            layer_dims=mlp_layer_dims,
            encoder_kwargs=encoder_kwargs,
        )

    def _get_output_shapes(self):
        """
        Allow subclasses to re-define outputs from @MIMO_MLP, since we won't
        always directly predict actions, but may instead predict the parameters
        of a action distribution.
        """
        return OrderedDict(action=(self.ac_dim,))

    def output_shape(self, input_shape=None):
        return [self.ac_dim]

    def forward(self, obs_dict):
        actions = super(ActorNetwork, self).forward(obs_dict)["action"]
        # apply tanh squashing to ensure actions are in [-1, 1]
        return torch.tanh(actions)

    def _to_string(self):
        """Info to pretty print."""
        return "action_dim={}".format(self.ac_dim)


class GMMActorNetwork(ActorNetwork):
    """
    Variant of actor network that learns a multimodal Gaussian mixture distribution
    over actions.
    """

    def __init__(
            self,
            obs_shapes,
            ac_dim,
            mlp_layer_dims,
            num_modes=5,
            min_std=0.01,
            std_activation="softplus",
            low_noise_eval=True,
            use_tanh=False,
            goal_shapes=None,
            encoder_kwargs=None,
    ):
        """
        Args:
            obs_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for observations.

            ac_dim (int): dimension of action space.

            mlp_layer_dims ([int]): sequence of integers for the MLP hidden layers sizes.

            num_modes (int): number of GMM modes

            min_std (float): minimum std output from network

            std_activation (None or str): type of activation to use for std deviation. Options are:

                `'softplus'`: Softplus activation applied

                `'exp'`: Exp applied; this corresponds to network output being interpreted as log_std instead of std

            low_noise_eval (float): if True, model will sample from GMM with low std, so that
                one of the GMM modes will be sampled (approximately)

            use_tanh (bool): if True, use a tanh-Gaussian distribution

            goal_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for goal observations.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """

        # parameters specific to GMM actor
        self.num_modes = num_modes
        self.min_std = min_std
        self.low_noise_eval = low_noise_eval
        self.use_tanh = use_tanh

        # Define activations to use
        self.activations = {
            "softplus": F.softplus,
            "exp": torch.exp,
        }
        assert std_activation in self.activations, \
            "std_activation must be one of: {}; instead got: {}".format(self.activations.keys(), std_activation)
        self.std_activation = std_activation

        super(GMMActorNetwork, self).__init__(
            obs_shapes=obs_shapes,
            ac_dim=ac_dim,
            mlp_layer_dims=mlp_layer_dims,
            goal_shapes=goal_shapes,
            encoder_kwargs=encoder_kwargs,
        )

    def _get_output_shapes(self):
        """
        Tells @MIMO_MLP superclass about the output dictionary that should be generated
        at the last layer. Network outputs parameters of GMM distribution.
        """
        return OrderedDict(
            mean=(self.num_modes, self.ac_dim),
            scale=(self.num_modes, self.ac_dim),
            logits=(self.num_modes,),
        )

    def forward_train(self, obs_dict):
        """
        Return full GMM distribution, which is useful for computing
        quantities necessary at train-time, like log-likelihood, KL
        divergence, etc.

        Args:
            obs_dict (dict): batch of observations
            goal_dict (dict): if not None, batch of goal observations

        Returns:
            dist (Distribution): GMM distribution
        """
        out = MIMO_MLP.forward(self, inputs=obs_dict)
        means = out["mean"]
        scales = out["scale"]
        logits = out["logits"]

        # apply tanh squashing to means if not using tanh-GMM to ensure means are in [-1, 1]
        if not self.use_tanh:
            means = torch.tanh(means)

        # Calculate scale
        if self.low_noise_eval and (not self.training):
            # low-noise for all Gaussian dists
            scales = torch.ones_like(means) * 1e-4
        else:
            # post-process the scale accordingly
            scales = self.activations[self.std_activation](scales) + self.min_std

        # mixture components - make sure that `batch_shape` for the distribution is equal
        # to (batch_size, num_modes) since MixtureSameFamily expects this shape
        component_distribution = D.Normal(loc=means, scale=scales)
        component_distribution = D.Independent(component_distribution, 1)

        # unnormalized logits to categorical distribution for mixing the modes
        mixture_distribution = D.Categorical(logits=logits)

        dist = D.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )

        if self.use_tanh:
            assert False
            # Wrap distribution with Tanh
            dist = TanhWrappedDistribution(base_dist=dist, scale=1.)

        return dist

    def forward(self, obs_dict):
        """
        Samples actions from the policy distribution.

        Args:
            obs_dict (dict): batch of observations
            goal_dict (dict): if not None, batch of goal observations

        Returns:
            action (torch.Tensor): batch of actions from policy distribution
        """
        dist = self.forward_train(obs_dict)
        return dist.sample()

    def _to_string(self):
        """Info to pretty print."""
        return "action_dim={}\nnum_modes={}\nmin_std={}\nstd_activation={}\nlow_noise_eval={}".format(
            self.ac_dim, self.num_modes, self.min_std, self.std_activation, self.low_noise_eval)

    def jacobian_train_gaussian(self, cond_dict, expert_model_gaussian: 'GaussianModel' = None):
        self.optimizer.zero_grad()

        self._jacobian_mode = True

        params = dict(self.named_parameters())
        jacobians, (predicted_action, predicted_means, predicted_log_stds) \
            = jacrev(functional_call, argnums=1, has_aux=True)(self, params, (cond_dict,))
        self._jacobian_mode = False

        predicted_distribution = Normal(predicted_means, predicted_log_stds.exp(), validate_args=False)
        log_probs = predicted_distribution.log_prob(predicted_action)

        batch_size = predicted_action.shape[0]
        with torch.no_grad():
            dist = expert_model_gaussian.forward(cond_dict)
            mean = dist.mean
            variance = dist.variance
            score = -1 * (predicted_action - mean) / variance

        for key, jacobian in jacobians.items():
            param = params[key]
            result = []

            dims = [0] + list(range(2, jacobian.dim())) + [1]
            jacobian = jacobian.permute(*dims)
            jacobian = jacobian.reshape(batch_size, -1, self.action_size)
            grad = -1 * torch.bmm(jacobian, score.unsqueeze(-1)).squeeze(-1)
            grad = grad.mean(dim=0).reshape(param.shape)
            param.grad = grad

        # Now the grad contains the -score * grad phi (a) so we have to add the grad phi logprob pi_phi(a)
        log_prob_loss = torch.mean(torch.sum(log_probs, dim=-1))
        log_prob_loss.backward(retain_graph=True)

        return predicted_action, log_probs


## UNIMODAL GAUSSIAN
class GaussianActorNetwork(ActorNetwork):
    """
    Variant of actor network that learns a diagonal unimodal Gaussian distribution
    over actions.
    """

    def __init__(
            self,
            obs_shapes,
            ac_dim,
            mlp_layer_dims,
            fixed_std=False,
            # std_activation="exp",
            init_last_fc_weight=None,
            init_std=0.3,
            # mean_limits=(-1_000_000, 9.0),
            std_limits=(-20, 2),
            low_noise_eval=True,
            use_tanh=False,
            goal_shapes=None,
            encoder_kwargs=None,
    ):
        """
        Args:
            obs_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for observations.

            ac_dim (int): dimension of action space.

            mlp_layer_dims ([int]): sequence of integers for the MLP hidden layers sizes.

            fixed_std (bool): if True, std is not learned, but kept constant at @init_std

            std_activation (None or str): type of activation to use for std deviation. Options are:

                None: no activation applied (not recommended unless using fixed std)

                `'softplus'`: Only applicable if not using fixed std. Softplus activation applied, after which the
                    output is scaled by init_std / softplus(0)

                `'exp'`: Only applicable if not using fixed std. Exp applied; this corresponds to network output
                    as being interpreted as log_std instead of std

                NOTE: In all cases, the final result is clipped to be within @std_limits

            init_last_fc_weight (None or float): if specified, will intialize the final layer network weights to be
                uniformly sampled from [-init_weight, init_weight]

            init_std (None or float): approximate initial scaling for standard deviation outputs
                from network. If None

            mean_limits (2-array): (min, max) to clamp final mean output by

            std_limits (2-array): (min, max) to clamp final std output by

            low_noise_eval (float): if True, model will output means of Gaussian distribution
                at eval time.

            use_tanh (bool): if True, use a tanh-Gaussian distribution

            goal_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for goal observations.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """

        # internal field
        self._jacobian_mode = False

        # parameters specific to Gaussian actor
        self.fixed_std = fixed_std
        self.init_std = init_std
        # self.mean_limits = np.array(mean_limits)
        self.std_limits = np.array(std_limits)

        # Define activations to use
        # def softplus_scaled(x):
        #     out = F.softplus(x)
        #     out = out * (self.init_std / F.softplus(torch.zeros(1).to(x.device)))
        #     return out

        # self.activations = {
        #     None: lambda x: x,
        #     # "softplus": softplus_scaled,
        #     "exp": torch.exp,
        # }
        # assert std_activation in self.activations, \
        #     "std_activation must be one of: {}; instead got: {}".format(self.activations.keys(), std_activation)
        # self.std_activation = std_activation if not self.fixed_std else None

        self.low_noise_eval = low_noise_eval
        self.use_tanh = use_tanh

        super(GaussianActorNetwork, self).__init__(
            obs_shapes=obs_shapes,
            ac_dim=ac_dim,
            mlp_layer_dims=mlp_layer_dims,
            goal_shapes=goal_shapes,
            encoder_kwargs=encoder_kwargs,
        )

        # If initialization weight was specified, make sure all final layer network weights are specified correctly
        if init_last_fc_weight is not None:
            with torch.no_grad():
                for name, layer in self.nets["decoder"].nets.items():
                    torch.nn.init.uniform_(layer.weight, -init_last_fc_weight, init_last_fc_weight)
                    torch.nn.init.uniform_(layer.bias, -init_last_fc_weight, init_last_fc_weight)

    def _get_output_shapes(self):
        """
        Tells @MIMO_MLP superclass about the output dictionary that should be generated
        at the last layer. Network outputs parameters of Gaussian distribution.
        """
        return OrderedDict(
            mean=(self.ac_dim,),
            scale=(self.ac_dim,),
        )

    def forward(self, obs_dict):
        """
        Return full Gaussian distribution, which is useful for computing
        quantities necessary at train-time, like log-likelihood, KL
        divergence, etc.

        Args:
            obs_dict (dict): batch of observations
            goal_dict (dict): if not None, batch of goal observations

        Returns:
            dist (Distribution): Gaussian distribution
        """
        out = MIMO_MLP.forward(self, inputs=obs_dict)
        mean = out["mean"]
        # Use either constant std or learned std depending on setting
        scale = out["scale"] if not self.fixed_std else torch.ones_like(mean) * self.init_std

        # Clamp the mean
        # mean = torch.clamp(mean, min=self.mean_limits[0], max=self.mean_limits[1])

        # apply tanh squashing to mean if not using tanh-Gaussian to ensure mean is in [-1, 1]
        if not self.use_tanh:
            mean = torch.tanh(mean)

        # Calculate scale
        if self.low_noise_eval and (not self.training):
            # override std value so that you always approximately sample the mean
            scale = torch.ones_like(mean) * 1e-4
        else:
            # Clamp the logstds:
            # Clamp the scale
            scale = torch.clamp(scale, min=self.std_limits[0], max=self.std_limits[1])

            # Post-process the scale accordingly
            scale = torch.exp(scale)

        # the Independent call will make it so that `batch_shape` for dist will be equal to batch size
        # while `event_shape` will be equal to action dimension - ensuring that log-probability
        # computations are summed across the action dimension

        dist = D.Normal(loc=mean, scale=scale, validate_args=False)
        dist = D.Independent(dist, 1, validate_args=False)
        if self._jacobian_mode:
            actions = dist.rsample()
            return actions, (actions, mean, scale)

        if self.use_tanh:
            assert False
            # Wrap distribution with Tanh
            dist = TanhWrappedDistribution(base_dist=dist, scale=1.)

        return dist

    # def forward(self, obs_dict):
    # """
    # Samples actions from the policy distribution.
    #
    # Args:
    #     obs_dict (dict): batch of observations
    #     goal_dict (dict): if not None, batch of goal observations
    #
    # Returns:
    #     action (torch.Tensor): batch of actions from policy distribution
    # """
    # dist = self.forward_train(obs_dict)
    # if self.low_noise_eval and (not self.training):
    #     if self.use_tanh:
    #         # # scaling factor lets us output actions like [-1. 1.] and is consistent with the distribution transform
    #         # return (1. + 1e-6) * torch.tanh(dist.base_dist.mean)
    #         return torch.tanh(dist.mean)
    #     return dist.mean
    # return dist.sample()
    # # return dist.mean

    def _to_string(self):
        """Info to pretty print."""
        msg = "action_dim={}\nfixed_std={}\nstd_activation={}\ninit_std={}\nmean_limits={}\nstd_limits={}\nlow_noise_eval={}".format(
            self.ac_dim, self.fixed_std, self.std_activation, self.init_std, self.mean_limits, self.std_limits,
            self.low_noise_eval)
        return msg

    def jacobian_train_diffusion(self, expert_diffusion_model, normalized_image_obs, expert_config: DiffusionModelRunConfig, beta):
        # Assumes that this model only takes one condition called "state" with the shape (B, obs_horizon * obs_dim)
        # And that it only predicts one action

        # self.optimizer.zero_grad()

        self._jacobian_mode = True
        cond_dict = {"observation": normalized_image_obs.flatten(start_dim=1)}

        params = dict(self.named_parameters())
        # Note the jacobians have gradient attached which could be removed somehow
        jacobians, (predicted_action, predicted_means, predicted_log_stds) \
            = jacrev(functional_call, argnums=1, has_aux=True)(self, params, (cond_dict,))
        self._jacobian_mode = False

        predicted_distribution = Normal(predicted_means, predicted_log_stds, validate_args=False)
        log_probs = predicted_distribution.log_prob(predicted_action)

        batch_size = predicted_action.shape[0]
        with torch.no_grad():
            # Calculate the score:
            score = expert_diffusion_model['noise_pred_net'](
                predicted_action.reshape(batch_size, expert_config.pred_horizon, -1), torch.zeros(batch_size, device="cuda"),
                global_cond=cond_dict["observation"]).squeeze(1)
            score = score / (- 1 * torch.sqrt(beta))
            score = score.flatten(start_dim=1)
        gradient_norm = 0
        for key, jacobian in jacobians.items():
            param = params[key]

            dims = [0] + list(range(2, jacobian.dim())) + [1]
            jacobian = jacobian.permute(*dims)
            jacobian = jacobian.reshape(batch_size, -1, expert_config.action_dim * expert_config.pred_horizon)
            grad = -1 * torch.bmm(jacobian, score.unsqueeze(-1)).squeeze(-1)
            grad = grad.mean(dim=0).reshape(param.shape)
            param.grad = grad
            gradient_norm += grad.norm().item()
        gradient_norm /= len(jacobians)

        # Now the grad contains the -score * grad phi (a) so we have to add the grad phi logprob pi_phi(a)
        log_prob_loss = torch.mean(torch.sum(log_probs, dim=-1))
        log_prob_loss.backward(retain_graph=True)

        return predicted_distribution
