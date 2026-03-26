# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
def plot_learning_curves(csv_file):
    # Load the data using Pandas
    df = pd.read_csv(csv_file)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot Training and Validation loss
    plt.plot(df['Epoch'], df['Train_Loss'], label='Training Loss', 
             color='blue', linewidth=2, marker='o', markersize=4)
    plt.plot(df['Epoch'], df['Val_Loss'], label='Validation Loss', 
             color='orange', linewidth=2, marker='s', markersize=4)
    
    # Formatting for maximum readability (good for papers/presentations)
    plt.title('CNN Training & Validation Loss', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Gaussian Negative Log-Likelihood Loss', fontsize=14)
    
    # Add a grid to make it easier to read values
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add the legend
    plt.legend(fontsize=12)
    
    # Ensure layout is tight so labels aren't cut off
    plt.tight_layout()
    
    # Save the plot as a high-res PNG
    output_filename = csv_file.replace('.csv', '_plot.png')
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved to {output_filename}")
    
    # Display the plot
    plt.show()

# Run the function
plot_learning_curves("training_log_v1_baseline.csv")