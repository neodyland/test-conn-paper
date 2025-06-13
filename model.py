import torch
import torch.nn as nn
import torch.nn.functional as F
import math

CONV_KERNEL_SIZE = 4
CONV_PADDING = 3
DEFAULT_CHUNK_SIZE = 64
EPSILON = 1e-9


def is_training(module):
    return module.training


class RoPE(nn.Module):
    def __init__(self, dimension, max_sequence_length=8192, base_frequency=10000.0):
        super().__init__()
        self.dimension = dimension
        self.max_sequence_length = max_sequence_length
        self.base_frequency = base_frequency

        inverse_frequencies = 1.0 / (
            self.base_frequency
            ** (torch.arange(0, self.dimension, 2).float() / self.dimension)
        )
        self.register_buffer("inverse_frequencies", inverse_frequencies)
        self.max_cached_length = 0
        self.cosine_cache = None
        self.sine_cache = None

    def _build_cache(self, sequence_length):
        if sequence_length > self.max_cached_length:
            self.max_cached_length = sequence_length
            time_steps = torch.arange(
                sequence_length, device=self.inverse_frequencies.device
            )
            frequencies = torch.einsum("i,j->ij", time_steps, self.inverse_frequencies)
            embeddings = torch.cat((frequencies, frequencies), dim=-1)
            self.cosine_cache = embeddings.cos()
            self.sine_cache = embeddings.sin()
        return self.cosine_cache[:sequence_length], self.sine_cache[:sequence_length]

    def _apply_rotary_embedding(self, tensor, cosine, sine):
        rotated_tensor = torch.cat(
            [-tensor[..., self.dimension // 2 :], tensor[..., : self.dimension // 2]],
            dim=-1,
        )
        return tensor * cosine + rotated_tensor * sine

    def forward(self, query, key):
        sequence_length = query.shape[1]
        cosine, sine = self._build_cache(sequence_length)
        rotated_query = self._apply_rotary_embedding(query, cosine, sine)
        rotated_key = self._apply_rotary_embedding(key, cosine, sine)
        return rotated_query, rotated_key


class GatedMLP(nn.Module):
    def __init__(self, model_dimension, feedforward_dimension):
        super().__init__()
        self.gate_projection = nn.Linear(
            model_dimension, feedforward_dimension, bias=False
        )
        self.output_projection = nn.Linear(
            feedforward_dimension, model_dimension, bias=False
        )
        self.up_projection = nn.Linear(
            model_dimension, feedforward_dimension, bias=False
        )

    def forward(self, input_tensor):
        gate_output = F.silu(self.gate_projection(input_tensor))
        up_output = self.up_projection(input_tensor)
        return self.output_projection(gate_output * up_output)


class YAADLayer(nn.Module):
    def __init__(
        self,
        model_dimension,
        head_dimension,
        state_dimension,
        low_rank_dimension=64,
        expansion_factor=2,
    ):
        super().__init__()
        self.model_dimension = model_dimension
        self.head_dimension = head_dimension
        self.state_dimension = state_dimension
        self.mlp_inner_dimension = state_dimension * expansion_factor

        self._initialize_projections(
            model_dimension, head_dimension, low_rank_dimension
        )
        self._initialize_conv_layer(head_dimension)
        self._initialize_norms(head_dimension)
        self._initialize_memory_parameters()

    def _initialize_projections(
        self, model_dimension, head_dimension, low_rank_dimension
    ):
        self.query_projection = nn.Linear(model_dimension, head_dimension, bias=False)
        self.key_projection = nn.Linear(model_dimension, head_dimension, bias=False)
        self.value_projection = nn.Linear(model_dimension, head_dimension, bias=False)
        self.output_projection = nn.Linear(head_dimension, model_dimension, bias=False)
        self.gate_projection = nn.Linear(model_dimension, model_dimension, bias=False)

        self.eta_projection = nn.Sequential(
            nn.Linear(model_dimension, low_rank_dimension, bias=False),
            nn.Linear(low_rank_dimension, head_dimension, bias=False),
        )
        self.delta_projection = nn.Sequential(
            nn.Linear(model_dimension, low_rank_dimension, bias=False),
            nn.Linear(low_rank_dimension, head_dimension, bias=False),
        )
        self.alpha_projection = nn.Sequential(
            nn.Linear(model_dimension, low_rank_dimension, bias=False),
            nn.Linear(low_rank_dimension, head_dimension, bias=False),
        )

    def _initialize_conv_layer(self, head_dimension):
        self.convolution = nn.Conv1d(
            in_channels=head_dimension,
            out_channels=head_dimension,
            kernel_size=CONV_KERNEL_SIZE,
            padding=CONV_PADDING,
            groups=head_dimension,
        )

    def _initialize_norms(self, head_dimension):
        self.query_norm = nn.LayerNorm(head_dimension)
        self.key_norm = nn.LayerNorm(head_dimension)

    def _initialize_memory_parameters(self):
        self.memory_weight1 = nn.Parameter(
            torch.empty(self.mlp_inner_dimension, self.head_dimension)
        )
        self.memory_bias1 = nn.Parameter(torch.empty(self.mlp_inner_dimension))
        self.memory_weight2 = nn.Parameter(
            torch.empty(self.head_dimension, self.mlp_inner_dimension)
        )
        self.memory_bias2 = nn.Parameter(torch.empty(self.head_dimension))
        self._reset_memory_parameters()

    def _reset_memory_parameters(self):
        for weight, bias in [
            (self.memory_weight1, self.memory_bias1),
            (self.memory_weight2, self.memory_bias2),
        ]:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(bias, -bound, bound)

    def _forward_memory(self, key_chunk, memory_state, return_intermediate=False):
        weight1, bias1, weight2, bias2 = memory_state
        intermediate = key_chunk @ weight1.transpose(-2, -1) + bias1.unsqueeze(1)
        activated = F.gelu(intermediate)
        output = activated @ weight2.transpose(-2, -1) + bias2.unsqueeze(1)

        if return_intermediate:
            return output, activated
        return output

    def _apply_convolution(self, tensors, sequence_length):
        return tuple(
            self.convolution(tensor.transpose(1, 2)).transpose(1, 2)[
                :, :sequence_length
            ]
            for tensor in tensors
        )

    def _initialize_memory_state(self, batch_size):
        return [
            self.memory_weight1.unsqueeze(0).expand(batch_size, -1, -1),
            self.memory_bias1.unsqueeze(0).expand(batch_size, -1),
            self.memory_weight2.unsqueeze(0).expand(batch_size, -1, -1),
            self.memory_bias2.unsqueeze(0).expand(batch_size, -1),
        ]

    def _extract_chunk(self, tensors, start_index, end_index):
        return tuple(tensor[:, start_index:end_index] for tensor in tensors)

    def _compute_memory_updates(
        self,
        memory_output,
        value_chunk,
        eta_chunk,
        delta_chunk,
        alpha_chunk,
        activated_intermediate,
    ):
        error = memory_output - value_chunk
        error_norm = torch.linalg.norm(error, dim=-1, keepdim=True)

        l2_gradient = error
        l1_gradient = delta_chunk * (error / (error_norm + EPSILON))
        prediction_gradient = torch.where(
            (error_norm > delta_chunk), l1_gradient, l2_gradient
        )

        alpha_weight = alpha_chunk[:, -1, :].mean(-1).view(-1, 1, 1)
        eta_weight = eta_chunk[:, -1, :].mean(-1).view(-1, 1, 1)
        update_signal = prediction_gradient.mean(1, keepdim=True)

        return alpha_weight, eta_weight, update_signal, prediction_gradient

    def _update_memory_weights(
        self,
        current_state,
        alpha_weight,
        eta_weight,
        update_signal,
        activated_intermediate,
    ):
        weight1, bias1, weight2, bias2 = current_state

        weight1_gradient = (
            activated_intermediate.mean(1, keepdim=True).transpose(-2, -1)
            @ update_signal
        )
        weight2_gradient = update_signal.transpose(
            -2, -1
        ) @ activated_intermediate.mean(1, keepdim=True)

        updated_weight1 = alpha_weight * weight1 - eta_weight * weight1_gradient
        updated_weight2 = alpha_weight * weight2 - eta_weight * weight2_gradient

        return updated_weight1, updated_weight2

    def _update_memory_biases(
        self,
        current_state,
        alpha_chunk,
        eta_chunk,
        prediction_gradient,
        updated_weight2,
    ):
        _, bias1, _, bias2 = current_state

        alpha_bias = alpha_chunk[:, -1, :]
        eta_bias = eta_chunk[:, -1, :]

        bias2_update = prediction_gradient.mean(1)
        bias1_update = (prediction_gradient @ updated_weight2).mean(1)

        updated_bias2 = alpha_bias * bias2 - eta_bias * bias2_update
        updated_bias1 = (
            alpha_bias.mean(-1, keepdim=True) * bias1
            - eta_bias.mean(-1, keepdim=True) * bias1_update
        )

        return updated_bias1, updated_bias2

    def _process_chunk(
        self,
        chunk_index,
        chunk_size,
        sequence_length,
        key,
        value,
        eta,
        delta,
        alpha,
        memory_state,
    ):
        start_index = chunk_index * chunk_size
        end_index = min((chunk_index + 1) * chunk_size, sequence_length)

        key_chunk, value_chunk = self._extract_chunk(
            [key, value], start_index, end_index
        )
        eta_chunk, delta_chunk, alpha_chunk = self._extract_chunk(
            [eta, delta, alpha], start_index, end_index
        )

        if is_training(self):
            memory_output, activated_intermediate = self._forward_memory(
                key_chunk, memory_state, return_intermediate=True
            )
            updated_state = self._update_memory_state(
                memory_output,
                value_chunk,
                eta_chunk,
                delta_chunk,
                alpha_chunk,
                activated_intermediate,
                memory_state,
            )
            return memory_output, updated_state
        else:
            memory_output = self._forward_memory(key_chunk, memory_state)
            return memory_output, memory_state

    def _update_memory_state(
        self,
        memory_output,
        value_chunk,
        eta_chunk,
        delta_chunk,
        alpha_chunk,
        activated_intermediate,
        current_state,
    ):
        with torch.no_grad():
            alpha_weight, eta_weight, update_signal, prediction_gradient = (
                self._compute_memory_updates(
                    memory_output,
                    value_chunk,
                    eta_chunk,
                    delta_chunk,
                    alpha_chunk,
                    activated_intermediate,
                )
            )

            updated_weight1, updated_weight2 = self._update_memory_weights(
                current_state,
                alpha_weight,
                eta_weight,
                update_signal,
                activated_intermediate,
            )

            updated_bias1, updated_bias2 = self._update_memory_biases(
                current_state,
                alpha_chunk,
                eta_chunk,
                prediction_gradient,
                updated_weight2,
            )

            return [updated_weight1, updated_bias1, updated_weight2, updated_bias2]

    def forward(self, input_tensor, chunk_size=DEFAULT_CHUNK_SIZE):
        batch_size, sequence_length, _ = input_tensor.shape

        query, key, value = (
            self.query_projection(input_tensor),
            self.key_projection(input_tensor),
            self.value_projection(input_tensor),
        )

        eta = F.softplus(self.eta_projection(input_tensor))
        delta = F.softplus(self.delta_projection(input_tensor))
        alpha = torch.sigmoid(self.alpha_projection(input_tensor))

        query, key, value = self._apply_convolution(
            [query, key, value], sequence_length
        )
        query, key = self.query_norm(query), self.key_norm(key)

        number_of_chunks = (sequence_length + chunk_size - 1) // chunk_size
        chunk_outputs = []
        current_memory_state = self._initialize_memory_state(batch_size)

        for chunk_index in range(number_of_chunks):
            chunk_output, current_memory_state = self._process_chunk(
                chunk_index,
                chunk_size,
                sequence_length,
                key,
                value,
                eta,
                delta,
                alpha,
                current_memory_state,
            )
            chunk_outputs.append(chunk_output)

        combined_output = torch.cat(chunk_outputs, dim=1)
        projected_output = self.output_projection(combined_output)
        gate_values = torch.sigmoid(self.gate_projection(input_tensor))

        return projected_output * gate_values


class YAADBlock(nn.Module):
    def __init__(
        self,
        model_dimension,
        head_dimension,
        state_dimension,
        feedforward_dimension,
        low_rank_dimension,
        expansion_factor,
    ):
        super().__init__()
        self.yaad_layer = YAADLayer(
            model_dimension,
            head_dimension,
            state_dimension,
            low_rank_dimension,
            expansion_factor,
        )
        self.mlp = GatedMLP(model_dimension, feedforward_dimension)
        self.first_norm = nn.RMSNorm(model_dimension)
        self.second_norm = nn.RMSNorm(model_dimension)

    def forward(self, input_tensor):
        attention_output = input_tensor + self.yaad_layer(self.first_norm(input_tensor))
        final_output = attention_output + self.mlp(self.second_norm(attention_output))
        return final_output


class YAADModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        number_of_layers,
        model_dimension,
        head_dimension,
        state_dimension,
        feedforward_dimension,
        low_rank_dimension,
        expansion_factor,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, model_dimension)
        self.transformer_layers = nn.ModuleList(
            [
                YAADBlock(
                    model_dimension,
                    head_dimension,
                    state_dimension,
                    feedforward_dimension,
                    low_rank_dimension,
                    expansion_factor,
                )
                for _ in range(number_of_layers)
            ]
        )
        self.output_norm = nn.RMSNorm(model_dimension)
        self.language_model_head = nn.Linear(model_dimension, vocab_size, bias=False)

    def forward(self, input_tokens):
        hidden_states = self.token_embedding(input_tokens)

        for transformer_layer in self.transformer_layers:
            hidden_states = transformer_layer(hidden_states)

        normalized_output = self.output_norm(hidden_states)
        logits = self.language_model_head(normalized_output)
        return logits

    def muon_parameters(self):
        params = [
            *self.transformer_layers.parameters(),
            *self.output_norm.parameters(),
        ]
        params = [p for p in params if p.requires_grad and p.dim() >= 2]
        return params

    def adam_parameters(self):
        params = [
            *self.transformer_layers.parameters(),
            *self.output_norm.parameters(),
        ]
        params = (
            [p for p in params if p.requires_grad and p.dim() < 2]
            + list(self.language_model_head.parameters())
            + list(self.token_embedding.parameters())
        )
        return params
