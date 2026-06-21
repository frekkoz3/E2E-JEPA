r"""
  _____ ____  _____          _ _____ ____   _    
 | ____|___ \| ____|        | | ____|  _ \ / \   
 |  _|   __) |  _| _____ _  | |  _| | |_) / _ \  
 | |___ / __/| |__|_____| |_| | |___|  __/ ___ \ 
 |_____|_____|_____|     \___/|_____|_| /_/   \_\
"""
from src.jepa.jepa import VisionTransformer, ActionConditionedPredictor
from src.policy.policy import Policy
from src.game.snake import SnakeEnv
from src.jepa.e2e_jepa import *

TOTAL_EPOCHS = 20000
STEPS_PER_EPOCH = 256
BATCH_SIZE = 32
ACTION_DIM = 4
REFRESH_BUFFER = 8

if __name__ == '__main__':
    
    trainer = ActiveE2EJEPATrainer(
        encoder=VisionTransformer(),
        predictor=ActionConditionedPredictor(),
        policy_network=PolicyNetwork(),
        action_dim=ACTION_DIM
    )
    
    env = SnakeEnv()
    x_t, _ = env.reset()
    
    for epoch in range(TOTAL_EPOCHS):
        if epoch % REFRESH_BUFFER == 0:
            trainer.register_buffer.refresh()
            
        for step in range(STEPS_PER_EPOCH):
            
            # Choose action actively using current model state
            a_t, _ = trainer.get_action(x_t)
            
            # Step the real environment
            x_tp1, r_t, done, _, _ = env.step(a_t)
            
            a_t_onehot = F.one_hot(torch.tensor(a_t), num_classes=ACTION_DIM).float()
            
            if done:
                x_tp1 = env.death_state()
                # Stream data into the experience buffer seamlessly
                trainer.buffer.push(x_t, a_t_onehot, r_t, x_tp1, float(done))
                x_t, _ = env.reset()
            else:
                # Stream data into the experience buffer seamlessly
                trainer.buffer.push(x_t, a_t_onehot, r_t, x_tp1, float(done))
                # Move forward
                x_t = x_tp1
        
        # Optimize over collected transitions at the end of the epoch step block
        metrics = trainer.update_parameters(BATCH_SIZE, epoch, TOTAL_EPOCHS)
        if metrics:
            print(f"Epoch {epoch} Metrics -> Loss: {metrics['total_loss']:.4f} | Pred: {metrics['pred_loss']:.4f}")
