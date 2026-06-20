"""
E2E-Jepa

Team Rocket:
@capsia37
@enricosavorgnan
@frekkoz3

The file implements a bunch of algorithms for Policy Learning, including:
  - Deep Q-Networks (DQN),
  - Convolutional DQN,
  - Attention DQN
"""

import torch
import torch.nn as nn




class DQN(nn.Module):
    """
    Deep Q_network for approximating Q-values.
    This is the standard PyTorch implementation. DO NOT MODIFY IT
    """

    def __init__(self,
                 input_dim : int,
                 output_dim : int,
                 num_hidden_layer : int,
                 dim_hidden_layer : list[int],
                 **kwargs):
        """
        Applies a feedforward neural network with ReLU activations to approximate Q-values.

        Parameters
        ----------
        input_dim : int
            Dimension of the input state.
        output_dim : int
            Dimension of the output Q-values (number of actions).
        num_hidden_layer : int
            Number of layers in the network.
            Does NOT include the input and output layers.
        dim_hidden_layer : list[int]
            List of hidden layer sizes.
            The length of this list must be equal to num_hidden_layer.
        """
        super(DQN, self).__init__()

        assert len(dim_hidden_layer) == num_hidden_layer, \
            f"The number of dimensions ({len(dim_hidden_layer)}) must match the number of layers ({num_hidden_layer})."

        self.first_layer = nn.Sequential(nn.Linear(input_dim, dim_hidden_layer[0]), nn.ReLU())
        self.hidden_layers = nn.Sequential(*
            [
            nn.Sequential(nn.Linear(dim_hidden_layer[i], dim_hidden_layer[i+1]), nn.ReLU())
            for i in range(num_hidden_layer-1)
            ],
        )
        self.output = nn.Linear(dim_hidden_layer[-1], output_dim)


    def forward(self, state):
        """Forward pass through the network."""
        state = self.first_layer(state)
        state = self.hidden_layers(state)
        q_values = self.output(state)
        return q_values



class ConvDQN(nn.Module):
    """
    Convolutional Deep Q-Network for approximating Q-values.
    The network applies 1D convolutional layers followed by fully connected layers to approximate Q-values.
    """

    def __init__(self,
                 input_dim : int,
                 output_dim : int,
                 num_conv_layer : int,
                 conv_layer_params : list[dict],
                 num_fc_layer : int,
                 dim_fc_layer : list[int],
                 **kwargs):
        """
        Initializes the ConvDQN network.

        Parameters
        ----------
        input_dim : int
            Dimension of the input state.
        output_dim : int
            Dimension of the output Q-values (number of actions).
        num_conv_layer : int
            Number of convolutional layers in the network.
        conv_layer_params : list[dict]
            List of dictionaries containing parameters for each convolutional layer.
            Each dictionary should contain 'out_channels', 'kernel_size', and 'stride'.
        num_fc_layer : int
            Number of fully connected layers in the network.
        dim_fc_layer : list[int]
            List of fully connected layer sizes.
            The length of this list must be equal to num_fc_layer.
        """
        super(ConvDQN, self).__init__()

        assert len(conv_layer_params) == num_conv_layer, \
            (f"The number of convolutional layer parameters ({len(conv_layer_params)}) "
             f"must match the number of convolutional layers ({num_conv_layer}).")
        assert len(dim_fc_layer) == num_fc_layer, \
            (f"The number of fully connected layer dimensions ({len(dim_fc_layer)}) "
             f"must match the number of fully connected layers ({num_fc_layer}).")

        self.conv_layers = nn.Sequential(*
            [
                nn.Sequential(
                    nn.Conv1d(in_channels=1 if i==0 else conv_layer_params[i-1].get('out_channels', 4),
                              out_channels=conv_layer_params[i].get('out_channels', 4),
                              kernel_size=conv_layer_params[i].get('kernel_size', 3),
                              stride=conv_layer_params[i].get('stride', 1)),
                    nn.ReLU()
                )
                for i in range(num_conv_layer)
            ]
        )
        self.fc_layers = nn.Sequential(*
            [
                nn.Sequential(
                    nn.Linear(dim_fc_layer[i], dim_fc_layer[i+1]),
                    nn.ReLU()
                )
                for i in range(num_fc_layer-1)
            ]
        )
        self.output = nn.Linear(dim_fc_layer[-1], output_dim)


    def forward(self, state):
        """ Forward pass through the network. """
        state = self.conv_layers(state)
        state = torch.flatten(state, start_dim=1)
        state = self.fc_layers(state)
        q_values = self.output(state)
        return q_values



class AttentionDQN(nn.Module):
    """
    Attention-based Deep Q-Network for approximating Q-values.
    The network applies attention mechanisms followed by fully connected layers to approximate Q-values.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim : int,
                 num_attention_layer : int,
                 attention_layer_params : list[dict],
                 num_fc_layer : int,
                 dim_fc_layer : list[int],
                 **kwargs):
        """
        Initializes the AttentionDQN network.

        Parameters
        ----------
        input_dim : int
            Dimension of the input state.
        output_dim : int
            Dimension of the output Q-values (number of actions).
        num_attention_layer : int
            Number of attention layers in the network.
        attention_layer_params : list[dict]
            List of dictionaries containing parameters for each attention layer.
            Each dictionary should contain 'num_heads' and 'dropout'.
        num_fc_layer : int
            Number of fully connected layers in the network.
        dim_fc_layer : list[int]
            List of fully connected layer sizes.
            The length of this list must be equal to num_fc_layer.
        """
        super(AttentionDQN, self).__init__()

        assert len(attention_layer_params) == num_attention_layer, \
            (f"The number of attention layer parameters ({len(attention_layer_params)}) "
             f"must match the number of attention layers ({num_attention_layer}).")
        assert len(dim_fc_layer) == num_fc_layer, \
            (f"The number of fully connected layer dimensions ({len(dim_fc_layer)}) "
             f"must match the number of fully connected layers ({num_fc_layer}).")

        # Transformer Encoder layers.
        attention_blocks = []
        for i in range(num_attention_layer):
            heads = attention_layer_params[i].get('num_heads', 4)

            assert input_dim % heads == 0, f"input_dim ({input_dim}) must be divisible by num_heads ({heads})."

            layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=heads,
                dim_feedforward=input_dim * 4, # Standard expansion factor
                activation='relu',
                batch_first=True # Expects [batch, seq, feature]
            )
            attention_blocks.append(layer)

        self.attention_layers = nn.Sequential(*attention_blocks)

        # 2. Fully Connected Head
        current_dim = input_dim
        self.fc_layers = nn.Sequential(*
            [
                nn.Sequential(
                    nn.Linear(current_dim if i==0 else dim_fc_layer[i-1], dim_fc_layer[i]),
                    nn.ReLU()
                )
                for i in range(num_fc_layer)
            ])

        self.output = nn.Linear(dim_fc_layer[-1], output_dim)


    def forward(self, state):
        """
        Forward pass through the network.
        state shape: [batch_size, sequence_length, input_dim]
        """
        # Output shape remains: [batch_size, sequence_length, input_dim]
        state = self.attention_layers(state)
        # Pooled shape: [batch_size, input_dim]
        state = state.mean(dim=1)

        state = self.fc_layers(state)
        q_values = self.output(state)

        return q_values



class ConvPPO(nn.Module):
    """ Convolutional Proximal Policy Optimization (PPO) network for approximating policy and value functions.
    The network applies 1D convolutional layers followed by fully connected layers to approximate both policy and value functions.
    """

    def __init__(self,
                 input_dim : int,
                 output_dim : int,
                 num_conv_layer : int,
                 conv_layer_params : list[dict],
                 num_fc_layer : int,
                 dim_fc_layer : list[int],
                 **kwargs):
        """
        Initializes the ConvPPO network.

        Parameters
        ----------
        input_dim : int
            Dimension of the input state.
        output_dim : int
            Dimension of the output Q-values (number of actions).
        num_conv_layer : int
            Number of convolutional layers in the network.
        conv_layer_params : list[dict]
            List of dictionaries containing parameters for each convolutional layer.
            Each dictionary should contain 'out_channels', 'kernel_size', and 'stride'.
        num_fc_layer : int
            Number of fully connected layers in the network.
        dim_fc_layer : list[int]
            List of fully connected layer sizes.
            The length of this list must be equal to num_fc_layer.
        """
        super(ConvPPO, self).__init__()

        assert len(conv_layer_params) == num_conv_layer, \
            (f"The number of convolutional layer parameters ({len(conv_layer_params)}) "
             f"must match the number of convolutional layers ({num_conv_layer}).")
        assert len(dim_fc_layer) == num_fc_layer, \
            (f"The number of fully connected layer dimensions ({len(dim_fc_layer)}) "
             f"must match the number of fully connected layers ({num_fc_layer}).")

        self.conv_layers = nn.Sequential(*
            [
                nn.Sequential(
                    nn.Conv1d(in_channels=1 if i==0 else conv_layer_params[i-1].get('out_channels', 4),
                              out_channels=conv_layer_params[i].get('out_channels', 4),
                              kernel_size=conv_layer_params[i].get('kernel_size', 3),
                              stride=conv_layer_params[i].get('stride', 1)),
                    nn.ReLU()
                )
                for i in range(num_conv_layer)
            ]
        )
        self.fc_layers = nn.Sequential(*
            [
                nn.Sequential(
                    nn.Linear(dim_fc_layer[i], dim_fc_layer[i+1]),
                    nn.ReLU()
                )
                for i in range(num_fc_layer-1)
            ]
        )

        # Dual heads
        self.actor_head = nn.Linear(dim_fc_layer[-1], output_dim)
        self.critic_head = nn.Linear(dim_fc_layer[-1], 1)


    def forward(self, state):
        """ Forward pass through the network. """
        state = self.conv_layers(state)
        state = torch.flatten(state, start_dim=1)
        state = self.fc_layers(state)

        logits = self.actor_head(state)
        value = self.critic_head(state)

        distribution = torch.distributions.Categorical(logits=logits)
        return distribution, value



class AttentionPPO(nn.Module):
    """
    Attention-based Proximal Policy Optimization (PPO) network for approximating policy and value functions.
    The network applies attention mechanisms followed by fully connected layers to approximate both policy and value functions.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim : int,
                 dim_stack : int,
                 num_attention_layer : int,
                 attention_layer_params : list[dict],
                 num_fc_layer : int,
                 dim_fc_layer : list[int],
                 **kwargs):
        """
        Initializes the AttentionDQN network.

        Parameters
        ----------
        input_dim : int
            Dimension of the input state.
        output_dim : int
            Dimension of the output Q-values (number of actions).
        num_attention_layer : int
            Number of attention layers in the network.
        attention_layer_params : list[dict]
            List of dictionaries containing parameters for each attention layer.
            Each dictionary should contain 'num_heads' and 'dropout'.
        num_fc_layer : int
            Number of fully connected layers in the network.
        dim_fc_layer : list[int]
            List of fully connected layer sizes.
            The length of this list must be equal to num_fc_layer.
        """
        super(AttentionPPO, self).__init__()

        assert len(attention_layer_params) == num_attention_layer, \
            (f"The number of attention layer parameters ({len(attention_layer_params)}) "
             f"must match the number of attention layers ({num_attention_layer}).")
        assert len(dim_fc_layer) == num_fc_layer, \
            (f"The number of fully connected layer dimensions ({len(dim_fc_layer)}) "
             f"must match the number of fully connected layers ({num_fc_layer}).")

        self.positional_encoding = nn.Parameter(torch.zeros(1, dim_stack, input_dim))
        nn.init.normal_(self.positional_encoding, std=0.02)

        # Transformer Encoder layers.
        attention_blocks = []
        for i in range(num_attention_layer):
            heads = attention_layer_params[i].get('num_heads', 4)

            assert input_dim % heads == 0, f"input_dim ({input_dim}) must be divisible by num_heads ({heads})."

            layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=heads,
                dim_feedforward=input_dim * 4, # Standard expansion factor
                activation='relu',
                batch_first=True # Expects [batch, seq, feature]
            )
            attention_blocks.append(layer)

        self.attention_layers = nn.Sequential(*attention_blocks)

        # 2. Fully Connected Head
        current_dim = input_dim
        self.fc_layers = nn.Sequential(*[
                                           nn.Sequential(
                                               nn.Linear(current_dim if i==0 else dim_fc_layer[i-1], dim_fc_layer[i]),
                                               nn.ReLU()
                                           )
                                           for i in range(num_fc_layer)
                                       ])

        # Dual heads
        self.actor_head = nn.Linear(dim_fc_layer[-1], output_dim)
        self.critic_head = nn.Linear(dim_fc_layer[-1], 1)


    def forward(self, state):
        """
        Forward pass through the network.
        state shape: [batch_size, sequence_length, input_dim]
        """
        state += self.positional_encoding[:, :state.size(1), :]

        state = self.attention_layers(state)

        # Extract last temporal step (the present frame)
        state = state[:, -1, :]

        state = self.fc_layers(state)

        logits = self.actor_head(state)
        value = self.critic_head(state)

        distribution = torch.distributions.Categorical(logits=logits)
        return distribution, value



if __name__ == "__main__":
    pass