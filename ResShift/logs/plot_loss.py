import re
import os
import numpy as np
import matplotlib.pyplot as plt

def parse_log_file(log_path):
    iterations = []
    mse_t1 = []
    lpips_t1 = []
    mse_t3 = []
    lpips_t3 = []
    mse_t4 = []
    lpips_t4 = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # Look for training lines
            if line.startswith('Train:'):
                # Extract iteration number
                iter_match = re.search(r'Train: (\d+)/350000', line)
                if iter_match:
                    iterations.append(int(iter_match.group(1)))
                
                # Extract loss values
                # Format: t(1):4.0e-02/8.2e-02, t(3):5.8e-01/2.6e-01, t(4):6.3e-01/2.8e-01
                losses = re.findall(r't\((\d)\):([\d.e-]+)/([\d.e-]+)', line)
                
                for t, mse, lpips in losses:
                    if t == '1':
                        mse_t1.append(float(mse))
                        lpips_t1.append(float(lpips))
                    elif t == '3':
                        mse_t3.append(float(mse))
                        lpips_t3.append(float(lpips))
                    elif t == '4':
                        mse_t4.append(float(mse))
                        lpips_t4.append(float(lpips))

    return {
        'iterations': np.array(iterations),
        'mse_t1': np.array(mse_t1),
        'lpips_t1': np.array(lpips_t1),
        'mse_t3': np.array(mse_t3),
        'lpips_t3': np.array(lpips_t3),
        'mse_t4': np.array(mse_t4),
        'lpips_t4': np.array(lpips_t4)
    }

def plot_losses(data, save_path):
    plt.figure(figsize=(12, 8))
    
    # Plot MSE losses
    plt.subplot(2, 1, 1)
    plt.plot(data['iterations'], data['mse_t1'], label='MSE t(1)')
    plt.plot(data['iterations'], data['mse_t3'], label='MSE t(3)')
    plt.plot(data['iterations'], data['mse_t4'], label='MSE t(4)')
    plt.xlabel('Iterations')
    plt.ylabel('MSE Loss')
    plt.title('MSE Losses vs Iterations')
    plt.grid(True)
    plt.legend()
    
    # Plot LPIPS losses
    plt.subplot(2, 1, 2)
    plt.plot(data['iterations'], data['lpips_t1'], label='LPIPS t(1)')
    plt.plot(data['iterations'], data['lpips_t3'], label='LPIPS t(3)')
    plt.plot(data['iterations'], data['lpips_t4'], label='LPIPS t(4)')
    plt.xlabel('Iterations')
    plt.ylabel('LPIPS Loss')
    plt.title('LPIPS Losses vs Iterations')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Get the directory of the log file
    log_path = 'logs/data_50k/2024-11-28-14-47/training.log'
    log_dir = os.path.dirname(log_path)
    
    # Parse the log file
    data = parse_log_file(log_path)
    
    # Create the plot and save it
    save_path = os.path.join(log_dir, 'training_losses.png')
    plot_losses(data, save_path)
    print(f"Plot saved to {save_path}")

if __name__ == '__main__':
    main()