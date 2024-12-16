import re
import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(log_path):
    # Initialize dictionaries to store losses
    losses = {
        'D_A': [], 'G_A': [], 'cycle_A': [], 'idt_A': [],
        'D_B': [], 'G_B': [], 'cycle_B': [], 'idt_B': []
    }
    epochs = []
    current_epoch = 0
    
    # Regular expression pattern to match loss values
    pattern = r'epoch: (\d+).+D_A: ([\d.]+) G_A: ([\d.]+) cycle_A: ([\d.]+) idt_A: ([\d.]+) D_B: ([\d.]+) G_B: ([\d.]+) cycle_B: ([\d.]+) idt_B: ([\d.]+)'
    
    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                if epoch != current_epoch:
                    current_epoch = epoch
                    epochs.append(epoch)
                    
                # Extract loss values
                losses['D_A'].append(float(match.group(2)))
                losses['G_A'].append(float(match.group(3)))
                losses['cycle_A'].append(float(match.group(4)))
                losses['idt_A'].append(float(match.group(5)))
                losses['D_B'].append(float(match.group(6)))
                losses['G_B'].append(float(match.group(7)))
                losses['cycle_B'].append(float(match.group(8)))
                losses['idt_B'].append(float(match.group(9)))
    
    return epochs, losses

def plot_losses(epochs, losses):
    plt.figure(figsize=(15, 10))
    
    # Plot discriminator losses
    plt.subplot(2, 2, 1)
    plt.plot(losses['D_A'], label='D_A', alpha=0.7)
    plt.plot(losses['D_B'], label='D_B', alpha=0.7)
    plt.title('Discriminator Losses')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot generator losses
    plt.subplot(2, 2, 2)
    plt.plot(losses['G_A'], label='G_A', alpha=0.7)
    plt.plot(losses['G_B'], label='G_B', alpha=0.7)
    plt.title('Generator Losses')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot cycle consistency losses
    plt.subplot(2, 2, 3)
    plt.plot(losses['cycle_A'], label='cycle_A', alpha=0.7)
    plt.plot(losses['cycle_B'], label='cycle_B', alpha=0.7)
    plt.title('Cycle Consistency Losses')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot identity losses
    plt.subplot(2, 2, 4)
    plt.plot(losses['idt_A'], label='idt_A', alpha=0.7)
    plt.plot(losses['idt_B'], label='idt_B', alpha=0.7)
    plt.title('Identity Losses')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('epoch_losses_50k.png')
    plt.close()

def main():
    log_path = 'logs/train_1359592.log'

    epochs, losses = parse_log_file(log_path)
    plot_losses(epochs, losses)
    print("Plots have been saved as 'epoch_losses_50k.png'")

if __name__ == '__main__':
    main() 