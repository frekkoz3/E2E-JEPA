import torch
from src.jepa.transformers import *

model = VisualTransformer(
    img_size=224,
    embed_dim=768,
    mlp_dim=3072,
    patch_size=16,
    num_heads=12,
    depth=6
)

x = torch.randn(2, 3, 224, 224)

y = model(x)

print(y.shape)

with torch.no_grad():
    patches = model.patch_embed(x)

    cls_before = model.cls_token.expand(x.size(0), -1, -1)

    tokens = torch.cat((cls_before, patches), dim=1)
    tokens = model.pos_embed(tokens)

    out = model.transformer(tokens)

    cls_after = out[:, 0]

print(torch.norm(cls_after - cls_before.squeeze(1)))

num_classes = 5

head = nn.Linear(768, num_classes)

images = torch.randn(10, 3, 224, 224)
labels = torch.randint(0, num_classes, (10,))

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(head.parameters()),
    lr=1e-4
)

for step in range(1000):
    optimizer.zero_grad()

    features = model(images)
    logits = head(features[:, 0])   # CLS token

    loss = F.cross_entropy(logits, labels)

    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        pred = logits.argmax(-1)
        acc = (pred == labels).float().mean()

        print(step, loss.item(), acc.item())