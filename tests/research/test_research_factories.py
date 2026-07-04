from veripet.research.heads import create_head
from veripet.research.losses import create_loss
from veripet.research.models import create_backbone


def test_backbone_head_and_loss_factories_cover_planned_grid():
    backbones = [
        "convnext_small.fb_in22k_ft_in1k",
        "efficientnet_b3.ra2_in1k",
        "swin_tiny_patch4_window7_224",
        "resnet101.a1_in1k",
    ]
    heads = ["Softmax", "ArcFace", "CosFace", "AdaFace", "MagFace"]
    losses = ["Softmax", "ArcFace", "CosFace", "AdaFace", "MagFace", "TripletMarginLoss"]

    assert all(create_backbone(name).name == name for name in backbones)
    assert all(create_head(name, embedding_dim=512, num_classes=8).name == name for name in heads)
    assert all(create_loss(name).name == name for name in losses)
