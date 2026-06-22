import torch
from src.jepa.transformers import *

"""
model = VisualTransformer(
    img_size=224,
    embed_dim=768,
    mlp_dim=3072,
    patch_size=16,
    num_heads=12,
    depth=6
)
"""
model = Transformer(
    input_dim = 10,
    hidden_dim = 5,
    output_dim = 5,
    depth = 3,
    num_heads = 1,
    mlp_dim = 6,
    dropout=0.0,
    use_adaLN=True,
)

device = "cuda"

model.to(device=device)

x = torch.randn(10, 10).to(device = device)
c = torch.randint(0, 4, (10, 10), dtype=torch.float32).to(device = device)

num_classes = 5

head = nn.Linear(5, num_classes)

head.to(device=device)

labels = torch.randint(0, num_classes, (10,)).to(device=device)

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(head.parameters()),
    lr=1e-4
)

for step in range(1000):
    optimizer.zero_grad()

    features = model(x, c)
    logits = head(features)   # CLS token

    loss = F.cross_entropy(logits, labels)

    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        pred = logits.argmax(-1)
        acc = (pred == labels).float().mean()

        print(step, loss.item(), acc.item())
