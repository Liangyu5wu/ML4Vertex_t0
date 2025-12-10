"""Model architecture modules.

Organized structure:
- common/: Shared components (base classes, layers, utilities)
- dnn/: DNN model implementations
- transformer/: Transformer model implementations
"""

# Import from reorganized structure
from .common import (
    BaseVertexModel,
    MaskedAttentionPooling,
    MaskedGlobalAveragePooling1D,
    FeatureEncoder,
    PositionalEncoding,
    MultiHeadSelfAttention,
    TransformerBlock
)

from .dnn import (
    DNNModel,
    BaselineGuidedDNN,
    MultiInputDNNModel,
    HGTDMultiInputDNNModel,
    HGTDOnlyDNNModel
)

from .transformer import (
    TransformerModel,
    MultiInputTransformerModel
)

__all__ = [
    # Common components
    'BaseVertexModel',
    'MaskedAttentionPooling',
    'MaskedGlobalAveragePooling1D',
    'FeatureEncoder',
    'PositionalEncoding',
    'MultiHeadSelfAttention',
    'TransformerBlock',
    # DNN models
    'DNNModel',
    'BaselineGuidedDNN',
    'MultiInputDNNModel',
    'HGTDMultiInputDNNModel',
    'HGTDOnlyDNNModel',
    # Transformer models
    'TransformerModel',
    'MultiInputTransformerModel'
]

