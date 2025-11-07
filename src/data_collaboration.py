from src.integrated_representation.build_integrated_representation import (
    IntegratedExpressionBuilder,
    IntegratedRepresentationBuilder,
)

# Backwards compatibility alias
DataCollaborationAnalysis = IntegratedExpressionBuilder

__all__ = ["IntegratedExpressionBuilder", "IntegratedRepresentationBuilder", "DataCollaborationAnalysis"]
