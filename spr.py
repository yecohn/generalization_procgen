import torch
from kornia.augmentation import RandomAffine,\
    RandomCrop,\
    CenterCrop, \
    RandomResizedCrop
from kornia.filters import GaussianBlur2d
from torch import nn
from utils import dummy_context_mgr, update_state_dict, infer_leading_dims
import numpy as np
import torch.nn.functional as F
import copy

class WorldModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            output_size,
            jumps,
            spr,
            augmentation,
            target_augmentation,
            eval_augmentation,
            dynamics_blocks,
            norm_type,
            noisy_nets,
            aug_prob,
            classifier,
            imagesize,
            time_offset,
            local_spr,
            global_spr,
            momentum_encoder,
            shared_encoder,
            momentum_tau,
            renormalize,
            q_l1_type,
            dropout,
            final_classifier,
            model_rl,
            residual_tm,
            use_maxpool=False,
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
            framestack=4,
            **kwargs
    ):
        """Instantiates the neural network according to arguments; network defaults
        stored within this method."""
        super().__init__()

        self.noisy = noisy_nets
        self.time_offset = time_offset
        self.aug_prob = aug_prob
        self.classifier_type = classifier


        self.transforms = []
        self.eval_transforms = []

        self.uses_augmentation = False
        for aug in augmentation:
            if aug == "affine":
                transformation = RandomAffine(5, (.14, .14), (.9, 1.1), (-5, 5))
                eval_transformation = nn.Identity()
                self.uses_augmentation = True
            elif aug == "crop":
                transformation = RandomCrop((64, 64))
                # Crashes if aug-prob not 1: use CenterCrop((84, 84)) or Resize((84, 84)) in that case.
                eval_transformation = CenterCrop((64, 64))
                self.uses_augmentation = True
                imagesize = 84
            elif aug == "rrc":
                transformation = RandomResizedCrop((100, 100), (0.8, 1))
                eval_transformation = nn.Identity()
                self.uses_augmentation = True
            elif aug == "blur":
                transformation = GaussianBlur2d((5, 5), (1.5, 1.5))
                eval_transformation = nn.Identity()
                self.uses_augmentation = True
            elif aug == "shift":
                transformation = nn.Sequential(nn.ReplicationPad2d(4), RandomCrop((64, 64)))
                eval_transformation = nn.Identity()
            elif aug == "intensity":
                transformation = Intensity(scale=0.05)
                eval_transformation = nn.Identity()
            elif aug == "none":
                transformation = eval_transformation = nn.Identity()
            else:
                raise NotImplementedError()
            self.transforms.append(transformation)
            self.eval_transforms.append(eval_transformation)

        f, c = image_shape[:2]
        in_channels = np.prod(image_shape[:2])
        self.conv = Conv2dModel(
            in_channels=in_channels,
            channels=[32, 64, 64],
            kernel_sizes=[8, 4, 3],
            strides=[4, 2, 1],
            paddings=[0, 0, 0],
            use_maxpool=False,
            dropout=dropout,
        )

        fake_input = torch.zeros(1, f*c, imagesize, imagesize)
        fake_output = self.conv(fake_input)
        self.hidden_size = fake_output.shape[1]
        self.pixels = fake_output.shape[-1]*fake_output.shape[-2]
        print("Spatial latent size is {}".format(fake_output.shape[1:]))

        self.jumps = jumps
        self.model_rl = model_rl
        self.use_spr = spr
        self.target_augmentation = target_augmentation
        self.eval_augmentation = eval_augmentation
        self.num_actions = output_size
        if self.jumps > 0:
            self.dynamics_model = TransitionModel(channels=self.hidden_size,
                                                  num_actions=output_size,
                                                  pixels=self.pixels,
                                                  hidden_size=self.hidden_size,
                                                  limit=1,
                                                  blocks=dynamics_blocks,
                                                  norm_type=norm_type,
                                                  renormalize=renormalize,
                                                  residual=residual_tm)
        else:
            self.dynamics_model = nn.Identity()

        self.renormalize = renormalize

        if self.use_spr:
            self.local_spr = local_spr
            self.global_spr = global_spr
            self.momentum_encoder = momentum_encoder
            self.momentum_tau = momentum_tau
            self.shared_encoder = shared_encoder
            assert not (self.shared_encoder and self.momentum_encoder)

            # in case someone tries something silly like --local-spr 2
            self.num_sprs = int(bool(self.local_spr)) + \
                            int(bool(self.global_spr))

            if self.local_spr:
                self.local_final_classifier = nn.Identity()
                if self.classifier_type == "mlp":
                    self.local_classifier = nn.Sequential(nn.Linear(self.hidden_size,
                                                                    self.hidden_size),
                                                          nn.BatchNorm1d(self.hidden_size),
                                                          nn.ReLU(),
                                                          nn.Linear(self.hidden_size,
                                                                    self.hidden_size))
                elif self.classifier_type == "bilinear":
                    self.local_classifier = nn.Linear(self.hidden_size, self.hidden_size)
                elif self.classifier_type == "none":
                    self.local_classifier = nn.Identity()
                if final_classifier == "mlp":
                    self.local_final_classifier = nn.Sequential(nn.Linear(self.hidden_size, 2*self.hidden_size),
                                                                nn.BatchNorm1d(2*self.hidden_size),
                                                                nn.ReLU(),
                                                                nn.Linear(2*self.hidden_size,
                                                                    self.hidden_size))
                elif final_classifier == "linear":
                    self.local_final_classifier = nn.Linear(self.hidden_size, self.hidden_size)
                else:
                    self.local_final_classifier = nn.Identity()

                self.local_target_classifier = self.local_classifier
            else:
                self.local_classifier = self.local_target_classifier = nn.Identity()
            if self.global_spr:
                self.global_final_classifier = nn.Identity()
                if self.classifier_type == "mlp":
                    self.global_classifier = nn.Sequential(
                                                nn.Flatten(-3, -1),
                                                nn.Linear(self.pixels*self.hidden_size, 512),
                                                nn.BatchNorm1d(512),
                                                nn.ReLU(),
                                                nn.Linear(512, 256)
                                                )
                    self.global_target_classifier = self.global_classifier
                    global_spr_size = 256
                elif self.classifier_type == "q_l1":
                    self.global_classifier = QL1Head(self.head, dueling=False, type=q_l1_type)
                    global_spr_size = self.global_classifier.out_features
                    self.global_target_classifier = self.global_classifier
                elif self.classifier_type == "q_l2":
                    self.global_classifier = nn.Sequential(self.head, nn.Flatten(-2, -1))
                    self.global_target_classifier = self.global_classifier
                    global_spr_size = 256
                elif self.classifier_type == "bilinear":
                    self.global_classifier = nn.Sequential(nn.Flatten(-3, -1),
                                                           nn.Linear(self.hidden_size*self.pixels,
                                                                     self.hidden_size*self.pixels))
                    self.global_target_classifier = nn.Flatten(-3, -1)
                elif self.classifier_type == "none":
                    self.global_classifier = nn.Flatten(-3, -1)
                    self.global_target_classifier = nn.Flatten(-3, -1)

                    global_spr_size = self.hidden_size*self.pixels
                if final_classifier == "mlp":
                    self.global_final_classifier = nn.Sequential(
                        nn.Linear(global_spr_size, global_spr_size*2),
                        nn.BatchNorm1d(global_spr_size*2),
                        nn.ReLU(),
                        nn.Linear(global_spr_size*2, global_spr_size)
                    )
                elif final_classifier == "linear":
                    self.global_final_classifier = nn.Sequential(
                        nn.Linear(global_spr_size, global_spr_size),
                    )
                elif final_classifier == "none":
                    self.global_final_classifier = nn.Identity()
            else:
                self.global_classifier = self.global_target_classifier = nn.Identity()

            if self.momentum_encoder:
                self.target_encoder = copy.deepcopy(self.conv)
                self.global_target_classifier = copy.deepcopy(self.global_target_classifier)
                self.local_target_classifier = copy.deepcopy(self.local_target_classifier)
                for param in (list(self.target_encoder.parameters())
                            + list(self.global_target_classifier.parameters())
                            + list(self.local_target_classifier.parameters())):
                    param.requires_grad = False

            elif not self.shared_encoder:
                # Use a separate target encoder on the last frame only.
                self.global_target_classifier = copy.deepcopy(self.global_target_classifier)
                self.local_target_classifier = copy.deepcopy(self.local_target_classifier)
                if self.stack_actions:
                    input_size = c - 1
                else:
                    input_size = c
                self.target_encoder = Conv2dModel(in_channels=input_size,
                                                  channels=[32, 64, 64],
                                                  kernel_sizes=[8, 4, 3],
                                                  strides=[4, 2, 1],
                                                  paddings=[0, 0, 0],
                                                  use_maxpool=False,
                                                  )

            elif self.shared_encoder:
                self.target_encoder = self.conv


    def spr_loss(self, f_x1s, f_x2s):
        f_x1 = F.normalize(f_x1s.float(), p=2., dim=-1, eps=1e-3)
        f_x2 = F.normalize(f_x2s.float(), p=2., dim=-1, eps=1e-3)
        # Gradients of norrmalized L2 loss and cosine similiarity are proportional.
        # See: https://stats.stackexchange.com/a/146279
        loss = F.mse_loss(f_x1, f_x2, reduction="none").sum(-1).mean(0)
        return loss

    def global_spr_loss(self, latents, target_latents, observation):
        global_latents = self.global_classifier(latents)
        global_latents = self.global_final_classifier(global_latents)
        with torch.no_grad() if self.momentum_encoder else dummy_context_mgr():
            global_targets = self.global_target_classifier(target_latents)
        targets = global_targets.view(-1, observation.shape[1],
                                             self.jumps+1, global_targets.shape[-1]).transpose(1, 2)
        latents = global_latents.view(-1, observation.shape[1],
                                             self.jumps+1, global_latents.shape[-1]).transpose(1, 2)
        loss = self.spr_loss(latents, targets)
        return loss

    def local_spr_loss(self, latents, target_latents, observation):
        local_latents = latents.flatten(-2, -1).permute(2, 0, 1)
        local_latents = self.local_classifier(local_latents)
        local_latents = self.local_final_classifier(local_latents)
        local_target_latents = target_latents.flatten(-2, -1).permute(2, 0, 1)
        with torch.no_grad() if self.momentum_encoder else dummy_context_mgr():
            local_targets = self.local_target_classifier(local_target_latents)

        local_latents = local_latents.view(-1,
                                           observation.shape[1],
                                           self.jumps+1,
                                           local_latents.shape[-1]).transpose(1, 2)
        local_targets = local_targets.view(-1,
                                           observation.shape[1],
                                           self.jumps+1,
                                           local_targets.shape[-1]).transpose(1, 2)
        local_loss = self.spr_loss(local_latents, local_targets)
        return local_loss

    def do_spr_loss(self, pred_latents, observation):
        pred_latents = torch.stack(pred_latents, 1)
        latents = pred_latents[:observation.shape[1]].flatten(0, 1)  # batch*jumps, *
        neg_latents = pred_latents[observation.shape[1]:].flatten(0, 1)
        latents = torch.cat([latents, neg_latents], 0)
        target_images = observation[self.time_offset:self.jumps + self.time_offset+1].transpose(0, 1).flatten(2, 3)
        target_images = self.transform(target_images, True)

        if not self.momentum_encoder and not self.shared_encoder:
            target_images = target_images[..., -1:, :, :]
        with torch.no_grad() if self.momentum_encoder else dummy_context_mgr():
            target_latents = self.target_encoder(target_images.flatten(0, 1))
            if self.renormalize:
                target_latents = renormalize(target_latents, -3)

        if self.local_spr:
            local_loss = self.local_spr_loss(latents, target_latents, observation)
        else:
            local_loss = 0
        if self.global_spr:
            global_loss = self.global_spr_loss(latents, target_latents, observation)
        else:
            global_loss = 0

        spr_loss = (global_loss + local_loss)/self.num_sprs
        spr_loss = spr_loss.view(-1, observation.shape[1]) # split to batch, jumps

        if self.momentum_encoder:
            update_state_dict(self.target_encoder,
                              self.conv,
                              self.momentum_tau)
            if self.classifier_type != "bilinear":
                # q_l1 is also bilinear for local
                if self.local_spr and self.classifier_type != "q_l1":
                    update_state_dict(self.local_target_classifier,
                                      self.local_classifier,
                                      self.momentum_tau)
                if self.global_spr:
                    update_state_dict(self.global_target_classifier,
                                      self.global_classifier,
                                      self.momentum_tau)
        return spr_loss

    def apply_transforms(self, transforms, eval_transforms, image):
        if eval_transforms is None:
            for transform in transforms:
                image = transform(image)
        else:
            for transform, eval_transform in zip(transforms, eval_transforms):
                image = maybe_transform(image, transform,
                                        eval_transform, p=self.aug_prob)
        return image

    @torch.no_grad()
    def transform(self, images, augment=False):
        images = images.float()/255. if images.dtype == torch.uint8 else images
        flat_images = images.reshape(-1, *images.shape[-3:])
        if augment:
            processed_images = self.apply_transforms(self.transforms,
                                                     self.eval_transforms,
                                                     flat_images)
        else:
            processed_images = self.apply_transforms(self.eval_transforms,
                                                     None,
                                                     flat_images)
        processed_images = processed_images.view(*images.shape[:-3],
                                                 *processed_images.shape[1:])
        return processed_images

    def stem_parameters(self):
        return list(self.conv.parameters()) + list(self.head.parameters())

    def stem_forward(self, img, prev_action=None, prev_reward=None):
        """Returns the normalized output of convolutional layers."""
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        conv_out = self.conv(img.view(T * B, *img_shape))  # Fold if T dimension.
        if self.renormalize:
            conv_out = renormalize(conv_out, -3)
        return conv_out

    def forward(self, observation,
                prev_action, prev_reward):
        """
        For convenience reasons with DistributedDataParallel the forward method
        has been split into two cases, one for training and one for eval.
        """
        pred_reward = []
        pred_latents = []
        input_obs = observation[0].flatten(1, 2)
        input_obs = self.transform(input_obs, augment=True)
        latent = self.stem_forward(input_obs,
                                    prev_action[0],
                                    prev_reward[0])
        pred_latents.append(latent)
        if self.jumps > 0:
            pred_rew = self.dynamics_model.reward_predictor(pred_latents[0])
            pred_reward.append(F.log_softmax(pred_rew, -1))

            for j in range(1, self.jumps + 1):
                latent, pred_rew = self.step(latent, prev_action[j])
                pred_rew = pred_rew[:observation.shape[1]]
                pred_latents.append(latent)
                pred_reward.append(F.log_softmax(pred_rew, -1))

        spr_loss = self.do_spr_loss(pred_latents, observation)
        return pred_reward, spr_loss

        

    def step(self, state, action):
        next_state, reward_logits = self.dynamics_model(state, action)
        return next_state, reward_logits

        
        
class Intensity(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise


class Conv2dModel(torch.nn.Module):
    """2-D Convolutional model component, with option for max-pooling vs
    downsampling for strides > 1.  Requires number of input channels, but
    not input shape.  Uses ``torch.nn.Conv2d``.
    """

    def __init__(
            self,
            in_channels,
            channels,
            kernel_sizes,
            strides,
            paddings=None,
            nonlinearity=torch.nn.ReLU,  # Module, not Functional.
            use_maxpool=False,  # if True: convs use stride 1, maxpool downsample.
            head_sizes=None,  # Put an MLP head on top.
            dropout=0.,
            ):
        super().__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [in_channels] + channels[:-1]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            maxp_strides = strides
            strides = ones
        else:
            maxp_strides = ones
        conv_layers = [torch.nn.Conv2d(in_channels=ic, out_channels=oc,
            kernel_size=k, stride=s, padding=p) for (ic, oc, k, s, p) in
            zip(in_channels, channels, kernel_sizes, strides, paddings)]
        sequence = list()
        for conv_layer, maxp_stride in zip(conv_layers, maxp_strides):
            sequence.extend([conv_layer, nonlinearity()])
            if dropout > 0:
                sequence.append(nn.Dropout(dropout))
            if maxp_stride > 1:
                sequence.append(torch.nn.MaxPool2d(maxp_stride))  # No padding.
        self.conv = torch.nn.Sequential(*sequence)

    def forward(self, input):
        """Computes the convolution stack on the input; assumes correct shape
        already: [B,C,H,W]."""
        return self.conv(input)


def init_normalization(channels, type="bn", affine=True, one_d=False):
    assert type in ["bn", "ln", "in", "none", None]
    if type == "bn":
        if one_d:
            return nn.BatchNorm1d(channels, affine=affine)
        else:
            return nn.BatchNorm2d(channels, affine=affine)
    elif type == "ln":
        if one_d:
            return nn.LayerNorm(channels, elementwise_affine=affine)
        else:
            return nn.GroupNorm(1, channels, affine=affine)
    elif type == "in":
        return nn.GroupNorm(channels, channels, affine=affine)
    elif type == "none" or type is None:
        return nn.Identity()


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_type="bn"):
        super().__init__()
        self.block = nn.Sequential(
            Conv2dSame(in_channels, out_channels, 3),
            nn.ReLU(),
            init_normalization(out_channels, norm_type),
            Conv2dSame(out_channels, out_channels, 3),
            init_normalization(out_channels, norm_type),
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = F.relu(out)
        return out


class Conv2dSame(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 stride=1,
                 padding_layer=nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias,
                            stride=stride, padding=ka)
        )

    def forward(self, x):
        return self.net(x)


def to_categorical(value, limit=300):
    value = value.float()  # Avoid any fp16 shenanigans
    value = value.clamp(-limit, limit)
    distribution = torch.zeros(value.shape[0], (limit*2+1), device=value.device)
    lower = value.floor().long() + limit
    upper = value.ceil().long() + limit
    upper_weight = value % 1
    lower_weight = 1 - upper_weight
    distribution.scatter_add_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
    distribution.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return distribution


def from_categorical(distribution, limit=300, logits=True):
    distribution = distribution.float()  # Avoid any fp16 shenanigans
    if logits:
        distribution = torch.softmax(distribution, -1)
    num_atoms = distribution.shape[-1]
    weights = torch.linspace(-limit, limit, num_atoms, device=distribution.device).float()
    return distribution @ weights


class TransitionModel(nn.Module):
    def __init__(self,
                 channels,
                 num_actions,
                 args=None,
                 blocks=16,
                 hidden_size=256,
                 pixels=36,
                 limit=300,
                 action_dim=6,
                 norm_type="bn",
                 renormalize=True,
                 residual=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.args = args
        self.renormalize = renormalize
        self.residual = residual
        layers = [Conv2dSame(channels+num_actions, hidden_size, 3),
                  nn.ReLU(),
                  init_normalization(hidden_size, norm_type)]
        for _ in range(blocks):
            layers.append(ResidualBlock(hidden_size,
                                        hidden_size,
                                        norm_type))
        layers.extend([Conv2dSame(hidden_size, channels, 3)])

        self.action_embedding = nn.Embedding(num_actions, pixels*action_dim)

        self.network = nn.Sequential(*layers)
        self.reward_predictor = RewardPredictor(channels,
                                                pixels=pixels,
                                                limit=limit,
                                                norm_type=norm_type)
        self.train()

    def forward(self, x, action):
        batch_range = torch.arange(action.shape[0], device=action.device)
        action_onehot = torch.zeros(action.shape[0],
                                    self.num_actions,
                                    x.shape[-2],
                                    x.shape[-1],
                                    device=action.device)
        action_onehot[batch_range, action, :, :] = 1
        stacked_image = torch.cat([x, action_onehot], 1)
        next_state = self.network(stacked_image)
        if self.residual:
            next_state = next_state + x
        next_state = F.relu(next_state)
        if self.renormalize:
            next_state = renormalize(next_state, 1)
        next_reward = self.reward_predictor(next_state)
        return next_state, next_reward


class RewardPredictor(nn.Module):
    def __init__(self,
                 input_channels,
                 hidden_size=1,
                 pixels=49,
                 limit=300,
                 norm_type="bn"):
        super().__init__()
        self.hidden_size = hidden_size
        layers = [nn.Conv2d(input_channels, hidden_size, kernel_size=1, stride=1),
                  nn.ReLU(),
                  init_normalization(hidden_size, norm_type),
                  nn.Flatten(-3, -1),
                  nn.Linear(pixels*hidden_size, 256),
                  nn.ReLU(),
                  nn.Linear(256, limit*2 + 1)]
        self.network = nn.Sequential(*layers)
        self.train()

    def forward(self, x):
        return self.network(x)


def renormalize(tensor, first_dim=1):
    if first_dim < 0:
        first_dim = len(tensor.shape) + first_dim
    flat_tensor = tensor.view(*tensor.shape[:first_dim], -1)
    max = torch.max(flat_tensor, first_dim, keepdim=True).values
    min = torch.min(flat_tensor, first_dim, keepdim=True).values
    flat_tensor = (flat_tensor - min)/(max - min)

    return flat_tensor.view(*tensor.shape)


class QL1Head(nn.Module):
    def __init__(self, head, dueling=False, type="noisy advantage"):
        super().__init__()
        self.head = head
        self.noisy = "noisy" in type
        self.dueling = dueling
        self.encoders = nn.ModuleList()
        self.relu = "relu" in type
        value = "value" in type
        advantage = "advantage" in type
        if self.dueling:
            if value:
                self.encoders.append(self.head.value[1])
            if advantage:
                self.encoders.append(self.head.advantage_hidden[1])
        else:
            self.encoders.append(self.head.network[1])

        self.out_features = sum([e.out_features for e in self.encoders])

    def forward(self, x):
        x = x.flatten(-3, -1)
        representations = []
        for encoder in self.encoders:
            encoder.noise_override = self.noisy
            representations.append(encoder(x))
            encoder.noise_override = None
        representation = torch.cat(representations, -1)
        if self.relu:
            representation = F.relu(representation)

        return representation

def maybe_transform(image, transform, alt_transform, p=0.8):
    processed_images = transform(image)
    if p >= 1:
        return processed_images
    else:
        base_images = alt_transform(image)
        mask = torch.rand((processed_images.shape[0], 1, 1, 1),
                          device=processed_images.device)
        mask = (mask < p).float()
        processed_images = mask * processed_images + (1 - mask) * base_images
        return processed_images