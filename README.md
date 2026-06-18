# E2E-JEPA

a project by Team Rocket - Bredariol, Riccio, Savorgnan

## The Idea

The **Joint Embedding Predictive Architecture (JEPA)** framework has recently emerged as a promising and appealing research direction in self-supervised learning. While studying the existing literature, our team began exploring a natural question: **can the JEPA paradigm be extended beyond representation learning and applied to more complex decision-making problems?**

The original JEPA framework focuses on learning predictive latent representations through self-supervised learning and does not include a policy-learning mechanism. Our goal is to investigate whether the latent space learned by a JEPA-style architecture can be leveraged as a compact state representation for policy learning, enabling the joint learning of both world representations and control policies in an end-to-end fashion.

To explore this idea, we adapt the basic JEPA framework and introduce some modifications aimed at integrating policy learning into the architecture. As an initial testbed, we focused on simple arcade-style video games. These environments offer a suitable balance between visual complexity and manageable dynamics, making them ideal for evaluating whether latent-space prediction can support effective decision making.

Our ultimate goal is to investigate whether a JEPA-based architecture can simultaneously learn meaningful world representations and a policy capable of acting within the environment, bringing together self-supervised predictive learning and policy learning in a unified framework.

## Repository Structure

```bash
├── configs/                  # Folder containing configs files
├── docs/                     # Folder containing paper references
├── src/                      # Folder containing the code base
└── README.md
```

## Formulation

## Quick Setup
