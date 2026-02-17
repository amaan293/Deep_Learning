import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)


def objective_function(x, y):
    return x**2 + y**2 + 10*np.sin(x) + 10*np.cos(y)


def compute_gradient(x, y):
    grad_x = 2*x + 10*np.cos(x)
    grad_y = 2*y - 10*np.sin(y)
    return grad_x, grad_y


def gradient_descent(x_init, y_init, learning_rate, num_iterations, tolerance=1e-6):
    x_trajectory = [x_init]
    y_trajectory = [y_init]
    f_trajectory = [objective_function(x_init, y_init)]
    grad_magnitude = []
    
    x_current = x_init
    y_current = y_init
    
    converged = False
    
    for iteration in range(num_iterations):
        grad_x, grad_y = compute_gradient(x_current, y_current)
        
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_magnitude.append(grad_mag)
        
        if grad_mag < tolerance:
            converged = True
            print(f"  Converged at iteration {iteration}")
            break
        
        x_new = x_current - learning_rate * grad_x
        y_new = y_current - learning_rate * grad_y
        
        x_trajectory.append(x_new)
        y_trajectory.append(y_new)
        f_trajectory.append(objective_function(x_new, y_new))
        
        x_current = x_new
        y_current = y_new
    
    if not converged:
        print(f"  Did not converge after {num_iterations} iterations")
    
    return {
        'x_trajectory': x_trajectory,
        'y_trajectory': y_trajectory,
        'f_trajectory': f_trajectory,
        'grad_magnitude': grad_magnitude,
        'converged': converged,
        'iterations': len(x_trajectory) - 1
    }


def create_contour_plot(ax, x_range, y_range, levels=30):
    x = np.linspace(x_range[0], x_range[1], 400)
    y = np.linspace(y_range[0], y_range[1], 400)
    X, Y = np.meshgrid(x, y)
    
    Z = objective_function(X, Y)
    
    contour = ax.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
    contourf = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.3)
    
    return contour, contourf


def plot_trajectory(ax, trajectory_dict, color, label):
    x_traj = trajectory_dict['x_trajectory']
    y_traj = trajectory_dict['y_trajectory']
    
    ax.plot(x_traj, y_traj, color=color, linewidth=2, label=label, alpha=0.8)
    
    ax.plot(x_traj[0], y_traj[0], marker='o', markersize=8, 
            color=color, markeredgecolor='black', markeredgewidth=1.5)
    
    ax.plot(x_traj[-1], y_traj[-1], marker='s', markersize=8, 
            color=color, markeredgecolor='black', markeredgewidth=1.5)


def experiment_learning_rate():
    print("\nEXPERIMENT 1: Effect of Learning Rate")
    print("Testing different learning rates from SAME starting point\n")
    
    num_iterations = 300
    x_init, y_init = 4.0, 4.0
    
    learning_rates = [0.01, 0.1, 0.3]
    colors = ['red', 'blue', 'green']
    
    results = []
    
    print(f"Fixed Starting Point: ({x_init}, {y_init})")
    print(f"Initial function value: {objective_function(x_init, y_init):.4f}\n")
    
    for lr, color in zip(learning_rates, colors):
        print(f"Learning Rate = {lr}")
        result = gradient_descent(x_init, y_init, lr, num_iterations)
        results.append((lr, result, color))
        
        final_x = result['x_trajectory'][-1]
        final_y = result['y_trajectory'][-1]
        final_f = result['f_trajectory'][-1]
        
        print(f"Final Position: ({final_x:.4f}, {final_y:.4f})")
        print(f"Final Function Value: {final_f:.4f}")
        print(f"Total Iterations: {result['iterations']}")
        print()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1 = axes[0]
    create_contour_plot(ax1, (-6, 6), (-6, 6))
    
    # Plot green first (bottom), then red and blue on top
    # Green thin and transparent, red and blue thick and solid
    for lr, result, color in results:
        if color == 'green':
            ax1.plot(result['x_trajectory'], result['y_trajectory'], 
                    color=color, linewidth=0.3, label=f'LR={lr}', alpha=0.4)
            ax1.plot(result['x_trajectory'][0], result['y_trajectory'][0], 
                    marker='o', markersize=8, color=color, 
                    markeredgecolor='black', markeredgewidth=1.5)
            ax1.plot(result['x_trajectory'][-1], result['y_trajectory'][-1], 
                    marker='s', markersize=8, color=color, 
                    markeredgecolor='black', markeredgewidth=1.5)
    
    for lr, result, color in results:
        if color != 'green':
            ax1.plot(result['x_trajectory'], result['y_trajectory'], 
                    color=color, linewidth=4, label=f'LR={lr}', alpha=1.0)
            ax1.plot(result['x_trajectory'][0], result['y_trajectory'][0], 
                    marker='o', markersize=10, color=color, 
                    markeredgecolor='black', markeredgewidth=2)
            ax1.plot(result['x_trajectory'][-1], result['y_trajectory'][-1], 
                    marker='s', markersize=10, color=color, 
                    markeredgecolor='black', markeredgewidth=2)
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Effect of Learning Rate', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    for lr, result, color in results:
        iterations = range(len(result['f_trajectory']))
        ax2.plot(iterations, result['f_trajectory'], color=color, 
                linewidth=2.5, label=f'LR={lr}')
    
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('f(x,y)', fontsize=12)
    ax2.set_title('Function Value vs Iteration', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results


def experiment_initial_position():
    print("\nEXPERIMENT 2: Effect of Initial Position")
    print("Testing different starting points with same learning rate\n")
    
    learning_rate = 0.1
    num_iterations = 300
    
    initial_positions = [
        (5.0, 5.0),
        (-5.0, 3.0),
        (1.0, -5.0),
    ]
    colors = ['red', 'blue', 'green']
    
    results = []
    
    print(f"Fixed Learning Rate: {learning_rate}\n")
    
    for (x_init, y_init), color in zip(initial_positions, colors):
        print(f"Starting Position: ({x_init}, {y_init})")
        print(f"Initial function value: {objective_function(x_init, y_init):.4f}")
        
        result = gradient_descent(x_init, y_init, learning_rate, num_iterations)
        results.append(((x_init, y_init), result, color))
        
        final_x = result['x_trajectory'][-1]
        final_y = result['y_trajectory'][-1]
        final_f = result['f_trajectory'][-1]
        
        print(f"Final Position: ({final_x:.4f}, {final_y:.4f})")
        print(f"Final Function Value: {final_f:.4f}")
        print(f"Total Iterations: {result['iterations']}")
        print()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1 = axes[0]
    create_contour_plot(ax1, (-6, 6), (-6, 6))
    
    for (x_init, y_init), result, color in results:
        plot_trajectory(ax1, result, color, f'({x_init:.0f}, {y_init:.0f})')
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Effect of Initial Position', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    for (x_init, y_init), result, color in results:
        iterations = range(len(result['f_trajectory']))
        ax2.plot(iterations, result['f_trajectory'], color=color, 
                linewidth=2.5, label=f'({x_init:.0f}, {y_init:.0f})')
    
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('f(x,y)', fontsize=12)
    ax2.set_title('Function Value vs Iteration', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results


def experiment_comprehensive():
    print("\nEXPERIMENT 3: Comprehensive Analysis")
    print("Testing all combinations of learning rates and starting points\n")
    
    learning_rates = [0.01, 0.1, 0.3]
    initial_positions = [(5.0, 5.0), (-5.0, 3.0), (1.0, -5.0)]
    
    for lr in learning_rates:
        print(f"\nAnalyzing Learning Rate = {lr}")
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        create_contour_plot(ax, (-6, 6), (-6, 6), levels=25)
        
        colors = ['red', 'blue', 'green']
        for (x_init, y_init), color in zip(initial_positions, colors):
            print(f"  Running from start point ({x_init:.0f}, {y_init:.0f})...")
            result = gradient_descent(x_init, y_init, lr, 300)
            
            x_traj = result['x_trajectory']
            y_traj = result['y_trajectory']
            
            ax.plot(x_traj, y_traj, color=color, linewidth=2, alpha=0.8,
                   label=f'Start: ({x_init:.0f}, {y_init:.0f})')
            ax.plot(x_traj[0], y_traj[0], 'o', color=color, markersize=9, 
                   markeredgecolor='black', markeredgewidth=1.5)
            ax.plot(x_traj[-1], y_traj[-1], 's', color=color, markersize=9,
                   markeredgecolor='black', markeredgewidth=1.5)
        
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(f'Learning Rate = {lr}: Different Starting Points', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    print("\nCompleted comprehensive analysis")


def plot_gradient_analysis(lr_results, init_results):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1 = axes[0]
    for lr, result, color in lr_results:
        grad_mag = result['grad_magnitude']
        iterations = range(len(grad_mag))
        ax1.plot(iterations, grad_mag, color=color, linewidth=2.5, 
                label=f'LR={lr}')
    
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Gradient Magnitude', fontsize=12)
    ax1.set_title('Gradient Magnitude: Learning Rate Effect', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2 = axes[1]
    for (x_init, y_init), result, color in init_results:
        grad_mag = result['grad_magnitude']
        iterations = range(len(grad_mag))
        ax2.plot(iterations, grad_mag, color=color, linewidth=2.5, 
                label=f'({x_init:.0f}, {y_init:.0f})')
    
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Gradient Magnitude', fontsize=12)
    ax2.set_title('Gradient Magnitude: Initial Position Effect', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.show()


def plot_3d_surface():
    print("\nCreating 3D Surface Visualization")
    
    fig = plt.figure(figsize=(14, 6))
    
    x = np.linspace(-6, 6, 200)
    y = np.linspace(-6, 6, 200)
    X, Y = np.meshgrid(x, y)
    Z = objective_function(X, Y)
    
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8,
                           linewidth=0, antialiased=True)
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('y', fontsize=11)
    ax1.set_zlabel('f(x,y)', fontsize=11)
    ax1.set_title('3D Surface Plot', fontsize=11, fontweight='bold')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)
    
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, Z, levels=30, cmap='viridis')
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('y', fontsize=11)
    ax2.set_title('Contour Plot', fontsize=11, fontweight='bold')
    fig.colorbar(contour, ax=ax2, aspect=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("\nGRADIENT DESCENT OPTIMIZATION")
    print("Objective: Minimize f(x,y) = x^2 + y^2 + 10sin(x) + 10cos(y)")
    print("\nStarting experiments...\n")
    
    lr_results = experiment_learning_rate()
    print("\nExperiment 1 complete. Showing plots...")
    
    init_results = experiment_initial_position()
    print("\nExperiment 2 complete. Showing plots...")
    
    experiment_comprehensive()
    print("\nExperiment 3 complete.")
    
    print("\nGenerating gradient magnitude analysis...")
    plot_gradient_analysis(lr_results, init_results)
    
    print("\nGenerating 3D surface plot...")
    plot_3d_surface()
    
    print("\nALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("Close all plot windows to exit.")