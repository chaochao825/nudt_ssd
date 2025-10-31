import torch
import os
from attacks.attacks import SSDAttacks

def main(args, cfg):
    """Main entry point for SSD detector"""
    
    if cfg.mode == 'adv':
        # Generate adversarial samples
        attacker = SSDAttacks(cfg)
        attacker.run_adv(args)
    elif cfg.mode == 'attack':
        # Evaluate model under attack
        attacker = SSDAttacks(cfg)
        attacker.run_attack(args)
    elif cfg.mode == 'defend':
        # Apply defense mechanisms
        from defends.defends import SSDDefends
        defender = SSDDefends(cfg)
        defender.run_defend(args)
    elif cfg.mode == 'train':
        # Train the model
        from train.trainer import SSDTrainer
        trainer = SSDTrainer(cfg)
        trainer.train()
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

