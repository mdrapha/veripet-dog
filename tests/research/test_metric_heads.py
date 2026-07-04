import pytest

torch = pytest.importorskip("torch")

from veripet.research.metric_heads import build_metric_head


@pytest.mark.parametrize("name", ["Softmax", "ArcFace", "CosFace", "AdaFace", "MagFace"])
def test_metric_heads_return_finite_logits_and_regularization(name):
    torch.manual_seed(7)
    embeddings = torch.randn(6, 8, requires_grad=True)
    norms = embeddings.detach().norm(dim=1, keepdim=True)
    labels = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)

    head = build_metric_head(name, embedding_dim=8, num_classes=3)
    output = head(embeddings, labels, norms)

    assert output.logits.shape == (6, 3)
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(output.regularization)

    loss = torch.nn.functional.cross_entropy(output.logits, labels) + output.regularization
    loss.backward()
    assert embeddings.grad is not None
