import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def parse_log_file(log_path):
    # Initialize dictionaries to store losses for each epoch
    epoch_losses = defaultdict(lambda: defaultdict(list))
    current_epoch = 0
    
    # Regular expression pattern to match loss values
    pattern = r'epoch: (\d+).+D_A: ([\d.]+) G_A: ([\d.]+) cycle_A: ([\d.]+) idt_A: ([\d.]+) D_B: ([\d.]+) G_B: ([\d.]+) cycle_B: ([\d.]+) idt_B: ([\d.]+)'
    
    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                # Store all losses for this epoch
                epoch_losses[epoch]['D_A'].append(float(match.group(2)))
                epoch_losses[epoch]['G_A'].append(float(match.group(3)))
                epoch_losses[epoch]['cycle_A'].append(float(match.group(4)))
                epoch_losses[epoch]['idt_A'].append(float(match.group(5)))
                epoch_losses[epoch]['D_B'].append(float(match.group(6)))
                epoch_losses[epoch]['G_B'].append(float(match.group(7)))
                epoch_losses[epoch]['cycle_B'].append(float(match.group(8)))
                epoch_losses[epoch]['idt_B'].append(float(match.group(9)))
    
    # Calculate mean losses for each epoch
    epochs = sorted(epoch_losses.keys())
    avg_losses = {
        'D_A': [], 'G_A': [], 'cycle_A': [], 'idt_A': [],
        'D_B': [], 'G_B': [], 'cycle_B': [], 'idt_B': []
    }
    
    for epoch in epochs:
        for loss_type in avg_losses.keys():
            avg_losses[loss_type].append(np.mean(epoch_losses[epoch][loss_type]))
    
    return epochs, avg_losses

def plot_losses(epochs, losses):
    plt.figure(figsize=(15, 10))
    
    # Plot discriminator losses
    plt.subplot(2, 2, 1)
    plt.plot(epochs, losses['D_A'], label='D_A', alpha=0.7)
    plt.plot(epochs, losses['D_B'], label='D_B', alpha=0.7)
    plt.title('Discriminator Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot generator losses
    plt.subplot(2, 2, 2)
    plt.plot(epochs, losses['G_A'], label='G_A', alpha=0.7)
    plt.plot(epochs, losses['G_B'], label='G_B', alpha=0.7)
    plt.title('Generator Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot cycle consistency losses
    plt.subplot(2, 2, 3)
    plt.plot(epochs, losses['cycle_A'], label='cycle_A', alpha=0.7)
    plt.plot(epochs, losses['cycle_B'], label='cycle_B', alpha=0.7)
    plt.title('Cycle Consistency Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot identity losses
    plt.subplot(2, 2, 4)
    plt.plot(epochs, losses['idt_A'], label='idt_A', alpha=0.7)
    plt.plot(epochs, losses['idt_B'], label='idt_B', alpha=0.7)
    plt.title('Identity Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('epoch_losses_50k.png', dpi=300)
    plt.close()

def main():
    log_path = 'logs/train_1359592.log'
    epochs, losses = parse_log_file(log_path)
    plot_losses(epochs, losses)
    print("Plots have been saved as 'epoch_losses_50k.png'")

if __name__ == '__main__':
    main() 