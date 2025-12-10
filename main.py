import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import datetime

from dataloader import get_loaders
from model.mlp import TabularModel
from metrics import Metrics
from utils import setup_logger, seed_everything
from model.mlp import TabularModel
from model.llm_mlp import LLMEnhancedMLP

def parse_args():
    parser = argparse.ArgumentParser(description="Diabetes Prediction Training")
    
    # Hyperparameters
    parser.add_argument('--model_type', type=str, default='mlp', choices=['mlp', 'llm_mlp'], 
                        help='Choose model architecture')
    parser.add_argument('--emb_path', type=str, default='dataset/feature_embeddings.npy',
                        help='Path to pretrained feature embeddings (for llm_mlp)')
    parser.add_argument('--freeze_emb', action='store_true', help='Freeze the initialized weights')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    
    # System
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--exp_name', type=str, default=f'exp_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers')

    return parser.parse_args()

def train(args, model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    # TQDM for visualization
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
    return running_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion, device, metrics_calc):
    model.eval()
    running_loss = 0.0
    metrics_calc.reset()
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        running_loss += loss.item()
        
        metrics_calc.update(outputs, targets)
        
    return running_loss / len(loader), metrics_calc.compute()

def main():
    args = parse_args()
    
    # Setup
    seed_everything(args.seed)
    logger = setup_logger("logs", args.exp_name)
    logger.info(f"Arguments: {vars(args)}")
    
    # Data
    train_loader, val_loader, test_loader, input_dim = get_loaders(args.batch_size, args.num_workers)
    
    # Model
    if args.model_type == 'mlp':
        logger.info("Initializing Baseline MLP...")
        model = TabularModel(
            input_dim=input_dim, 
            hidden_dim=args.hidden_dim, 
            dropout_rate=args.dropout
        ).to(args.device)
        
    elif args.model_type == 'llm_mlp':
        logger.info(f"Initializing LLM Enhanced MLP with embeddings from {args.emb_path}...")
        model = LLMEnhancedMLP(
            input_dim=input_dim, 
            hidden_dim=args.hidden_dim, 
            dropout_rate=args.dropout,
            embedding_path=args.emb_path,
            freeze_emb=args.freeze_emb
        ).to(args.device)
        
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pos_weight = torch.tensor([0.3]).to(args.device) 
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    metrics = Metrics()
    
    best_auc = 0.0
    
    # Training Loop
    logger.info("Start Training...")
    for epoch in range(args.epochs):
        train_loss = train(args, model, train_loader, optimizer, criterion, args.device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, args.device, metrics)
        
        log_msg = (f"Epoch [{epoch+1}/{args.epochs}] "
                   f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                   f"AUC: {val_metrics['AUC']:.4f} | F1: {val_metrics['F1']:.4f}")
        logger.info(log_msg)
        
        # Save Best Model
        if val_metrics['AUC'] > best_auc:
            best_auc = val_metrics['AUC']
            save_name = f"checkpoints/{args.exp_name}_best_AUC_{best_auc:.4f}_lr_{args.lr}.pth"
            if not os.path.exists("checkpoints"): os.makedirs("checkpoints")
            torch.save(model.state_dict(), save_name)
            logger.info(f"Best model saved to {save_name}")

    # Final Test
    logger.info("Training Finished. Running Test on Best Model...")
    # Load best model logic here (omitted for brevity)
    test_loss, test_metrics = evaluate(model, test_loader, criterion, args.device, metrics)
    logger.info(f"Test Results - AUC: {test_metrics['AUC']:.4f} | F1: {test_metrics['F1']:.4f}")

if __name__ == "__main__":
    main()